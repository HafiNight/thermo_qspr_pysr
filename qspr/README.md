QSPR model and final aggregated models of QSPR + SR optimal outputs
-	Models & testmodels: preliminary models initialization
-	svr_optimize and xgb_optimize: optimization of hyperparameters of xgboost & svr
-	wholeDB_apply_xgb: 
o	Apply XGB predictions to get completed DB (choose to or not to overwrite experimental values – Full Pred or Pred + Exp)
o	Evaluate on ‘best’ equations from each run
o	Evaluate on every equations from PySR output
