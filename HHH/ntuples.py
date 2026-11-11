"""Local paths and batch settings for the existing HHH ntuple analysis."""

from argparse import ArgumentParser
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "FCCAnalyses/hhh/HHH_ntuples.py"
spec = importlib.util.spec_from_file_location("hhh_ntuple_base", BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Analysis(module.Analysis):
    def __init__(self, cmdline_args):
        super().__init__(cmdline_args)
        parser = ArgumentParser(description=__doc__)
        parser.add_argument("--production-dir", type=Path,
                            default=HERE.parent / "EventProducer")
        parser.add_argument("--process", default="mgp8_pp_hhh_84TeV")
        parser.add_argument("--threads", type=int, default=2)
        parser.add_argument("--queue", default="workday")
        remaining = [arg for arg in cmdline_args.get("remaining", []) if arg != "--"]
        args = parser.parse_args(remaining)

        self.input_dir = str(args.production_dir.resolve() / "DelphesEvents/fcc_v07/II")
        files = sorted((Path(self.input_dir) / args.process).glob("events_*.root"))
        if not files:
            raise FileNotFoundError(f"No reconstructed files for {args.process} in {self.input_dir}")
        self.process_list = {args.process: {"output": args.process, "chunks": len(files)}}
        self.output_dir = "ntuples"  # Relative to the submission directory, as FCCAnalyses requires.
        self.test_file = str(files[0])
        self.n_threads = args.threads
        self.batch_queue = args.queue
        self.user_batch_config = str(HERE / "condor.sub")
        self.is_signal = args.process == "mgp8_pp_hhh_84TeV"

    def output(self):
        branches = super().output()
        # Only HHH has the nine reweights. RDataFrame leaves unused definitions unevaluated.
        if not self.is_signal:
            branches = [branch for branch in branches if not branch.startswith("weight_k3")]
        return branches
