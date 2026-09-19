"""
Counterfactual citation audit for explainable deepfake detectors.  Each
module runs as a script: `python -m ccaudit.<module> --help`.

  m1_verify         pairing gate                     -> pairs.json
  m2_parse          face parsing and crops           -> parsed/index.json
  m3_splice         the splice operator              (library; used in memory by m5)
  m4_detectors      detector zoo                     (controls, VLMs, CNN with Grad-CAM)
  m5_runner         resumable sharded audit          -> raw_*.json, cache_*.json
  m6_metrics        AUC, FS, CR and related scores   -> metrics.json
  m7_report         HTML report and SVG figures      -> report.html
  m8_human_study    annotator materials
  m9_train_cnn      CNN baseline on the DEV split
  m10_localization  citation correctness from paired frames
  m11_proxy         deployable proxy validation
  m12_text_regions  free text -> region distribution
  m13_cset          causally supervised explanation tuning
  m14_attribution   gradient citations and encoder blind-spot probe
"""

__version__ = "0.1.0"
VERSION = __version__
