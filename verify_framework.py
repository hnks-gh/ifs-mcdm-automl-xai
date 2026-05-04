"""
Comprehensive Verification and Audit Script for IFS-MCDM-AutoML-XAI Framework
================================================================================

This script performs a complete audit of the framework, verifying:
1. MCDM outputs (weights, rankings, analysis)
2. ML outputs (imputation, forecasting, SHAP)
3. Mathematical correctness
4. Data integrity
5. Output generation status
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.exceptions import *
from core.data_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


class FrameworkAudit:
    """Comprehensive framework audit and verification."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.config = load_config()
        self.audit_report = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "summary": {},
            "issues": [],
            "warnings": []
        }
        
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete audit across all phases."""
        logger.info("=" * 80)
        logger.info("🔍 STARTING COMPREHENSIVE FRAMEWORK AUDIT")
        logger.info("=" * 80)
        
        # Phase 1: MCDM Audit
        logger.info("\n📋 Phase 1: MCDM Pipeline Audit")
        logger.info("-" * 80)
        self.audit_mcdm()
        
        # Phase 2: ML Data Integrity Audit
        logger.info("\n📋 Phase 2: ML Data Integrity Audit")
        logger.info("-" * 80)
        self.audit_ml_inputs()
        
        # Phase 3: ML Outputs Audit (if available)
        logger.info("\n📋 Phase 3: ML Outputs Audit")
        logger.info("-" * 80)
        self.audit_ml_outputs()
        
        # Phase 4: Mathematical Correctness Verification
        logger.info("\n📋 Phase 4: Mathematical Correctness Verification")
        logger.info("-" * 80)
        self.verify_mathematical_soundness()
        
        # Phase 5: Data Integrity Checks
        logger.info("\n📋 Phase 5: Data Integrity Verification")
        logger.info("-" * 80)
        self.verify_data_integrity()
        
        # Generate summary
        self.generate_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ AUDIT COMPLETE")
        logger.info("=" * 80)
        
        return self.audit_report
    
    def audit_mcdm(self) -> None:
        """Audit MCDM pipeline outputs."""
        mcdm_dir = self.output_dir / "mcdm"
        
        if not mcdm_dir.exists():
            logger.warning("⚠️  MCDM output directory not found")
            self.audit_report["issues"].append("MCDM directory missing")
            return
        
        try:
            # Check weights
            weights_dir = mcdm_dir / "weights"
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.csv"))
                logger.info(f"✅ Found {len(weight_files)} weight files")
                
                # Verify weight structure
                if weight_files:
                    sample_weight = pd.read_csv(weight_files[0], index_col=0)
                    logger.info(f"   - Weight shape: {sample_weight.shape}")
                    logger.info(f"   - Weight columns: {list(sample_weight.columns)[:5]}...")
                    
                    # Verify weights sum to 1
                    if "Weight" in sample_weight.columns:
                        weight_sum = sample_weight["Weight"].sum()
                        logger.info(f"   - Weight sum check: {weight_sum:.6f} (should ≈ 1.0)")
                        if abs(weight_sum - 1.0) > 0.01:
                            self.audit_report["warnings"].append(
                                f"Weight sum deviation: {weight_sum}"
                            )
                    
                    self.audit_report["phases"]["mcdm_weights"] = {
                        "status": "✅ PASS",
                        "files": len(weight_files),
                        "sample_shape": sample_weight.shape
                    }
            else:
                logger.warning("   ⚠️  Weights directory not found")
                self.audit_report["issues"].append("MCDM weights not generated")
            
            # Check rankings
            rankings_dir = mcdm_dir / "rankings"
            if rankings_dir.exists():
                ranking_files = list(rankings_dir.glob("*.csv"))
                logger.info(f"✅ Found {len(ranking_files)} ranking files")
                
                # Parse ranking files by method
                methods = {}
                for rf in ranking_files:
                    method = rf.stem.split("_")[2] if "_" in rf.stem else "unknown"
                    methods[method] = methods.get(method, 0) + 1
                
                logger.info(f"   - Methods found: {list(methods.keys())}")
                logger.info(f"   - Rankings per method: {methods}")
                
                # Verify ranking structure
                if ranking_files:
                    sample_ranking = pd.read_csv(ranking_files[0], index_col=0)
                    logger.info(f"   - Ranking shape: {sample_ranking.shape}")
                    logger.info(f"   - Ranking columns: {list(sample_ranking.columns)[:5]}...")
                    
                    # Check rank values are between 1 and n
                    if "Rank" in sample_ranking.columns:
                        min_rank = sample_ranking["Rank"].min()
                        max_rank = sample_ranking["Rank"].max()
                        logger.info(f"   - Rank range: [{min_rank}, {max_rank}]")
                        if min_rank < 1 or max_rank > len(sample_ranking):
                            self.audit_report["warnings"].append(
                                f"Invalid rank values: [{min_rank}, {max_rank}]"
                            )
                    
                    self.audit_report["phases"]["mcdm_rankings"] = {
                        "status": "✅ PASS",
                        "files": len(ranking_files),
                        "methods": methods,
                        "sample_shape": sample_ranking.shape
                    }
            else:
                logger.warning("   ⚠️  Rankings directory not found")
                self.audit_report["issues"].append("MCDM rankings not generated")
            
            # Check analysis
            analysis_dir = mcdm_dir / "analysis"
            if analysis_dir.exists():
                analysis_files = list(analysis_dir.glob("*"))
                logger.info(f"✅ Found {len(analysis_files)} analysis files")
                self.audit_report["phases"]["mcdm_analysis"] = {
                    "status": "✅ PASS",
                    "files": len(analysis_files)
                }
            else:
                logger.info("   ℹ️  Analysis directory not yet created (expected)")
                
        except Exception as e:
            logger.error(f"❌ MCDM audit error: {e}")
            self.audit_report["issues"].append(f"MCDM audit error: {str(e)}")
    
    def audit_ml_inputs(self) -> None:
        """Audit ML input data (imputed panel)."""
        try:
            ml_dir = self.output_dir / "ml"
            imputed_dir = ml_dir / "imputed"
            
            if imputed_dir.exists():
                imputed_files = list(imputed_dir.glob("*.parquet"))
                logger.info(f"✅ Found {len(imputed_files)} imputed data files")
                
                if imputed_files:
                    # Load and verify imputed data
                    imputed_df = pd.read_parquet(imputed_files[0])
                    logger.info(f"   - Imputed panel shape: {imputed_df.shape}")
                    logger.info(f"   - Columns: {list(imputed_df.columns)[:10]}...")
                    
                    # Check for NaN values
                    nan_count = imputed_df.isnull().sum().sum()
                    logger.info(f"   - NaN count after imputation: {nan_count}")
                    if nan_count > 0:
                        self.audit_report["warnings"].append(
                            f"Imputed data still contains {nan_count} NaN values"
                        )
                    
                    # Verify expected shape (882 rows x 31 columns)
                    expected_rows = len(self.config.data.years) * self.config.data.n_provinces
                    if imputed_df.shape[0] == expected_rows:
                        logger.info(f"   ✅ Panel row count correct: {expected_rows}")
                    else:
                        self.audit_report["warnings"].append(
                            f"Panel has {imputed_df.shape[0]} rows, expected {expected_rows}"
                        )
                    
                    self.audit_report["phases"]["ml_imputation"] = {
                        "status": "✅ PASS",
                        "shape": imputed_df.shape,
                        "nan_count": int(nan_count)
                    }
            else:
                logger.warning("   ⚠️  Imputed data directory not found")
                self.audit_report["warnings"].append("Imputed panel not yet generated")
                
        except Exception as e:
            logger.error(f"❌ ML input audit error: {e}")
            self.audit_report["issues"].append(f"ML input audit error: {str(e)}")
    
    def audit_ml_outputs(self) -> None:
        """Audit ML output data (forecasts, SHAP, rankings)."""
        try:
            ml_dir = self.output_dir / "ml"
            
            # Check forecasts
            forecasts_dir = ml_dir / "forecasts"
            if forecasts_dir.exists():
                forecast_files = list(forecasts_dir.glob("*.csv"))
                logger.info(f"✅ Found {len(forecast_files)} forecast files")
                
                if forecast_files:
                    sample_forecast = pd.read_csv(forecast_files[0], index_col=0)
                    logger.info(f"   - Forecast shape: {sample_forecast.shape}")
                    logger.info(f"   - Columns: {list(sample_forecast.columns)}")
                    
                    self.audit_report["phases"]["ml_forecasting"] = {
                        "status": "✅ PASS",
                        "files": len(forecast_files),
                        "sample_shape": sample_forecast.shape
                    }
            else:
                logger.info("   ℹ️  Forecasts directory (expected after training)")
            
            # Check SHAP results
            shap_dir = ml_dir / "shap"
            if shap_dir.exists():
                shap_files = list(shap_dir.glob("*.parquet")) + list(shap_dir.glob("*.csv"))
                logger.info(f"✅ Found {len(shap_files)} SHAP result files")
                self.audit_report["phases"]["ml_shap"] = {
                    "status": "✅ PASS",
                    "files": len(shap_files)
                }
            else:
                logger.info("   ℹ️  SHAP results not yet generated (expected during training)")
            
            # Check 2025 rankings
            rankings_2025_dir = ml_dir / "rankings_2025"
            if rankings_2025_dir.exists():
                ranking_files = list(rankings_2025_dir.glob("*.csv"))
                logger.info(f"✅ Found {len(ranking_files)} 2025 ranking files")
                self.audit_report["phases"]["ml_rankings_2025"] = {
                    "status": "✅ PASS",
                    "files": len(ranking_files)
                }
            else:
                logger.info("   ℹ️  2025 rankings not yet generated (expected after forecasting)")
                
        except Exception as e:
            logger.error(f"❌ ML outputs audit error: {e}")
            self.audit_report["issues"].append(f"ML outputs audit error: {str(e)}")
    
    def verify_mathematical_soundness(self) -> None:
        """Verify mathematical correctness of computations."""
        try:
            logger.info("Checking mathematical correctness...")
            
            mcdm_dir = self.output_dir / "mcdm"
            weights_dir = mcdm_dir / "weights"
            
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.csv"))
                
                if weight_files:
                    checks_passed = 0
                    checks_failed = 0
                    
                    for wf in weight_files[:3]:  # Sample check first 3 files
                        try:
                            df = pd.read_csv(wf, index_col=0)
                            
                            if "Weight" in df.columns:
                                weight_sum = df["Weight"].sum()
                                if abs(weight_sum - 1.0) < 0.01:
                                    checks_passed += 1
                                    logger.info(f"   ✅ {wf.name}: weights sum = {weight_sum:.6f}")
                                else:
                                    checks_failed += 1
                                    logger.warning(f"   ❌ {wf.name}: weights sum = {weight_sum:.6f} (expected 1.0)")
                        except Exception as e:
                            logger.error(f"   ❌ Error checking {wf.name}: {e}")
                            checks_failed += 1
                    
                    self.audit_report["phases"]["mathematical_verification"] = {
                        "status": "✅ PASS" if checks_failed == 0 else "⚠️  PARTIAL",
                        "checks_passed": checks_passed,
                        "checks_failed": checks_failed
                    }
            else:
                logger.info("   ℹ️  MCDM weights not yet available")
                
        except Exception as e:
            logger.error(f"❌ Mathematical verification error: {e}")
            self.audit_report["issues"].append(f"Mathematical verification error: {str(e)}")
    
    def verify_data_integrity(self) -> None:
        """Verify data integrity across pipeline."""
        try:
            logger.info("Checking data integrity...")
            
            integrity_checks = {
                "province_count": self.config.data.n_provinces,
                "year_count": len(self.config.data.years),
                "subcriteria_count": self.config.data.n_subcriteria
            }
            
            logger.info(f"   ✅ Expected config: {integrity_checks}")
            
            # Check MCDM outputs
            mcdm_dir = self.output_dir / "mcdm"
            weights_dir = mcdm_dir / "weights"
            
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.csv"))
                expected_years = len(self.config.data.years)
                actual_years = len(weight_files)
                
                if actual_years == expected_years:
                    logger.info(f"   ✅ Weight files: {actual_years}/{expected_years} years")
                else:
                    logger.warning(f"   ❌ Weight files: {actual_years}/{expected_years} years")
                    self.audit_report["issues"].append(
                        f"Weight files mismatch: {actual_years}/{expected_years}"
                    )
            
            # Check rankings
            rankings_dir = mcdm_dir / "rankings"
            if rankings_dir.exists():
                ranking_files = list(rankings_dir.glob("*.csv"))
                expected_rankings = len(self.config.data.years) * 3  # 3 methods
                actual_rankings = len(ranking_files)
                
                if actual_rankings == expected_rankings:
                    logger.info(f"   ✅ Ranking files: {actual_rankings}/{expected_rankings} files")
                else:
                    logger.warning(f"   ❌ Ranking files: {actual_rankings}/{expected_rankings} files")
                    self.audit_report["issues"].append(
                        f"Ranking files mismatch: {actual_rankings}/{expected_rankings}"
                    )
            
            self.audit_report["phases"]["data_integrity"] = {
                "status": "✅ PASS",
                "config": integrity_checks
            }
            
        except Exception as e:
            logger.error(f"❌ Data integrity verification error: {e}")
            self.audit_report["issues"].append(f"Data integrity error: {str(e)}")
    
    def generate_summary(self) -> None:
        """Generate audit summary."""
        total_issues = len(self.audit_report["issues"])
        total_warnings = len(self.audit_report["warnings"])
        
        self.audit_report["summary"] = {
            "total_phases_checked": len(self.audit_report["phases"]),
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "status": "✅ PASS" if total_issues == 0 else "⚠️  PARTIAL" if total_warnings == 0 else "❌ FAIL"
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 AUDIT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Phases checked: {self.audit_report['summary']['total_phases_checked']}")
        logger.info(f"Issues found: {total_issues}")
        logger.info(f"Warnings: {total_warnings}")
        logger.info(f"Overall status: {self.audit_report['summary']['status']}")
        
        if self.audit_report["issues"]:
            logger.warning("\n⚠️  ISSUES:")
            for issue in self.audit_report["issues"]:
                logger.warning(f"   - {issue}")
        
        if self.audit_report["warnings"]:
            logger.warning("\n⚠️  WARNINGS:")
            for warning in self.audit_report["warnings"]:
                logger.warning(f"   - {warning}")
        
        # Save audit report as JSON
        report_path = self.output_dir / "audit_report.json"
        with open(report_path, "w") as f:
            json.dump(self.audit_report, f, indent=2, default=str)
        logger.info(f"\n✅ Audit report saved to: {report_path}")


def main():
    """Run the comprehensive audit."""
    try:
        audit = FrameworkAudit()
        report = audit.run_full_audit()
        
        # Return exit code based on status
        if report["summary"]["status"] == "✅ PASS":
            sys.exit(0)
        elif report["summary"]["status"] == "⚠️  PARTIAL":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Fatal error during audit: {e}", exc_info=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
