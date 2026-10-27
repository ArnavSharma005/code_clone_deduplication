# Java semantic clone detection (GraphCodeBERT + FAISS + tree-sitter + ONNX INT8)

## Results (BigCloneBench, CodeXGLUE Java split, threshold tuned on validation)
| Model | Valid F1 | Test P / R / F1 |
|---|---|---|
| Pretrained GraphCodeBERT, cosine | 0.263 | 0.204 / 0.358 / 0.260 |
| Contrastive fine-tuned (2 epochs, 30k pairs) | 0.899 | 0.857 / 0.905 / **0.881** |
| INT8 ONNX vs fp32 (300-function test subset) | | F1 0.951 vs 0.953 (no meaningful loss) |

Reference: pairwise cross-encoder GraphCodeBERT reaches 0.950 F1 (no retrieval/FAISS). INT8 is 125MB vs 496MB.
INT8 CPU speed is ~0.9 s/function; Python imports take ~30 s on this machine, so each hook run is slow.

## Scripts (run from project root with `.venv\Scripts\python.exe`)
1. `scripts/01_prepare_data.py` - dataset -> `data/`
2. `scripts/02_baseline_eval.py` - pretrained baseline
3. `scripts/03_finetune_contrastive.py` - fine-tune (~100 min on RTX 3050 4GB), saves `models/finetuned/`
4. `scripts/04_export_onnx.py` - ONNX + INT8 export (`models/onnx/`)
5. `scripts/05_int8_eval.py 300` - INT8 vs fp32 accuracy on a subset

## Claude Code hook
Index a Java project once (re-run after big changes):
```
.venv\Scripts\python.exe hook\build_index.py C:\path\to\java\project
```
Add to that project's `.claude/settings.json`:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "E:\\projects\\code_clone_deduplication\\.venv\\Scripts\\python.exe E:\\projects\\code_clone_deduplication\\hook\\check_duplicates.py",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```
The hook embeds methods (>= 4 lines) in the edited `.java` file, searches the FAISS index, and exits 2 with a
message when similarity >= `CLONE_THRESHOLD` (default 0.82), which Claude sees and acts on.
Tested: a renamed-variable duplicate scored 0.85, an unrelated method 0.68.
