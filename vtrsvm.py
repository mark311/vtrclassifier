import sys,os
import json
import subprocess
import tempfile
from optparse import OptionParser

def find(dirpath, ext=None):
    if ext:
        ext = "." + ext
    for r, ds, fs in os.walk(dirpath):
        for f in fs:
            if ext == None or f.endswith(ext):
                yield os.path.join(r, f)

def filename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def parse_vt_rules(vtr_file):
    report = json.loads(open(vtr_file, "r").read())
    rules = []
    for product, detection in report["report"].items():
        if detection[0] != None:
            r = ("%s:%s" % (product, detection[0])).encode("UTF8")
            rules.append(r)
    return filename_without_ext(vtr_file), rules.__repr__()

def unix_slash(path):
    return path.replace("\\", "/")

def init_vtrsvm_workplace(sample_dir, vt_report_dir, vtr_cache_file,
                          vt_report_ext="txt"):
    # extract rules from all vt report files
    vtrules = {}
    for vtfile in find(vt_report_dir, ext=vt_report_ext):
        filehash, rules = parse_vt_rules(vtfile)
        vtrules[filehash] = rules

    # remap from filehash=>vtrules to samplefile=>vtrules
    cache_file = open(vtr_cache_file, "w")
    for samplefile in find(sample_dir):
        print >>cache_file, "%s, %s" % ( \
            unix_slash(samplefile),
            vtrules[filename_without_ext(samplefile)])
    cache_file.close()


class Sample:
    def __init__(self, filepath=None, classid=0, vt_rulenames=None):
        self.filepath = filepath
        self.classid = classid
        self.classid_predict = None
        self.vt_rulenames = vt_rulenames
        self.probability = 0.0

class SampleLoader:
    def __init__(self, vt_rulename_file):
        self.rulename = {}
        for line in open(vt_rulename_file, "r"):
            sp = line.split(",", 1)
            sample_file = sp[0].strip()
            rn = eval(sp[1].strip())
            self.rulename[sample_file] = rn

    def load_from_file(self, sample_list_file):
        sample_set = []
        for line in open(sample_list_file, "r"):
            if line.startswith("#"): continue
            line = line.strip()
            if line == "": continue
            sp = line.split(",")
            sample = Sample()
            sample.filepath = sp[0].strip()
            if len(sp) >= 2:
                sample.classid = int(sp[1])
            
            if sample.filepath in self.rulename:
                sample.vt_rulenames = self.rulename[sample.filepath]
            else:
                print "No rule name cache for sample %s" % sample.filepath

            sample_set.append(sample)

        return sample_set


class VTR_SVM_Processor:
    def __init__(self, model_file):
        self.svm_model_file = os.path.join(model_file, "model")
        self.ruleindex_file = os.path.join(model_file, "ruleid")
        self.feature_count = 0
        self.ruleindex = None

    def load_ruleindex(self):
        self.feature_count = 0
        self.ruleindex = {}

        f = open(self.ruleindex_file, "r")
        for line in f:
            line = line.strip()
            if line == "":
                continue
            sp = line.split(",")
            assert len(sp) == 2
            self.ruleindex[sp[1].strip()] = int(sp[0])
            self.feature_count += 1
        f.close()

    def save_ruleindex(self):
        f = open(self.ruleindex_file, "w")
        for rule, index in self.ruleindex.items():
            print >>f, "%d,%s" % (index, rule)
        f.close()

    def build_ruleindex(self, sample_set):
        self.ruleindex = {}
        self.feature_count = 0
        for sample in sample_set:
            for rulename in sample.vt_rulenames:
                if not rulename in self.ruleindex:
                    self.ruleindex[rulename] = self.feature_count
                    self.feature_count += 1
        self.save_ruleindex()

    def _sample2svm(self, sample, svm_input_file):
        v = [0,] * self.feature_count
        for rulename in sample.vt_rulenames:
            if rulename in self.ruleindex:
                v[self.ruleindex[rulename]] = 1

        svm_input_file.write("%s " % sample.classid)
        for i in range(self.feature_count):
            svm_input_file.write("%d:%d " % (i, v[i]))
        svm_input_file.write("\n")

    def _sampleset2svm(self, sample_set):
        fsvm = tempfile.NamedTemporaryFile(delete=False)
        for sample in sample_set:
            self._sample2svm(sample, fsvm)
        fsvm.close()
        return fsvm.name

    def train(self, training_set):
        self.build_ruleindex(training_set)
        svm_train_file = self._sampleset2svm(training_set)
        subprocess.call(["svm-train.exe", "-b", "1",
                         svm_train_file, self.svm_model_file])
        os.unlink(svm_train_file)

    def predict(self, predict_set, foutput):
        if not self.ruleindex:
            self.load_ruleindex()

        ftemp = tempfile.NamedTemporaryFile(delete=False)
        ftemp.close()

        svm_predict_file = self._sampleset2svm(predict_set)
        subprocess.call(["svm-predict.exe", "-b", "1",
                         svm_predict_file, self.svm_model_file,
                         ftemp.name])

        in_body = False
        i = 0
        for line in open(ftemp.name, "r"):
            if in_body:
                sp = line.split(" ")
                assert len(sp) >= 2
                predict_set[i].classid_predict = int(sp[0])
                predict_set[i].probability = 0.0
                for p in sp[1:]:
                    if float(p) > predict_set[i].probability:
                        predict_set[i].probability = float(p)
                i += 1
            else:
                if line.startswith("labels"):
                    in_body = True

        os.unlink(svm_predict_file)
        os.unlink(ftemp.name)

        for sample in predict_set:
            print >>foutput, sample.filepath, sample.classid_predict,\
                sample.probability

usage = '''
  %prog [options] sample_list
  %prog --train [options] sample_list
'''

def main():
    # command line parser
    op = OptionParser(usage)
    op.add_option("-m", "--model", dest="model",
                  default="vtrmodel",
                  help="A directory that saves model related files, "+
                  "./vtrmodel will be used by default.")
    op.add_option("-t", "--train", dest="train",
                  action="store_true", default=False,
                  help="To train the model instead of predict.")
    op.add_option("-s", "--sample", dest="sample",
                  help="The directory of sample files")
    op.add_option("-r", "--report", dest="report",
                  help="The directory of VirusTotal report files. The "+
                  "file name (without EXT) should be the same as the "+
                  "name of corresponding sample file.")
    op.add_option("-o", "--output", dest="output",
                  help="The file where the result will be written. If "+
                  "omitted, result will be written to stdout.")
    options, args = op.parse_args()

    if len(args) != 1:
        op.error("incorrect number of arguments")

    if options.sample == None:
        op.error("missing option -s,--sample")

    if options.report == None:
        op.error("missing option -r,--report")

    # parsing_cache_file is consist of options.report and ".vtr"
    if options.report.endswith("/") or options.report.endswith("\\"):
        parsing_cache_file = options.report[:-1] + ".vtr"
    else:
        parsing_cache_file = options.report + ".vtr"

    if not os.path.exists(parsing_cache_file):
        print >>sys.stderr, "No VirusTotal rule parsing cache file "+ \
            "found, creating it now ..."
        init_vtrsvm_workplace(options.sample, options.report,
                              parsing_cache_file, "txt")
        print >>sys.stderr, "%s created successfully." % parsing_cache_file

    # output
    if options.output == None:
        foutput = sys.stdout
    else:
        foutput = open(options.output, "w")

    # load sample set according to the first argument.
    loader = SampleLoader(parsing_cache_file)
    sample_set = loader.load_from_file(args[0])

    processor = VTR_SVM_Processor(options.model)
    if options.train:
        processor.train(sample_set)
    else:
        processor.predict(sample_set, foutput)

    return 0

if __name__ == "__main__":
    sys.exit(main())
