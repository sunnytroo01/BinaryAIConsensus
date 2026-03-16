"""Wipe network volume clean. Run before fresh training."""
import shutil, os
for d in ['/workspace/checkpoints', '/workspace/training_data']:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Deleted {d}")
    else:
        print(f"  {d} (already clean)")
print("  Network volume wiped. Ready for fresh start.")
