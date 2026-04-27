## Plan: Knitwear Recommender Accuracy + Footprint Upgrade

Prioritize recommendation accuracy first with retraining-enabled improvements, while reducing inference waste and deployment size in parallel. Execute low-risk latency and reliability fixes first, then improve retrieval quality with multi-vector indexing and better crop robustness, and finally prepare a smaller deployable runtime for flexible hosting.

**Steps**
1. Phase 1: Baseline and instrumentation. Add request-level timing and quality diagnostics in the API flow to measure crop success rate, embedding time, retrieval latency, and metadata fetch latency. This blocks all later phases because it defines before/after impact.
2. Phase 1: Remove redundant inference work by reusing the first crop/embedding outputs across recommendation and XAI paths. Depends on step 1.
3. Phase 1: Introduce lazy model loading for DINO and YOLO models with thread-safe singleton initialization to reduce cold-start startup pressure. Parallel with step 2.
4. Phase 1: Harden request handling with image-size validation, safer crop fallback thresholds, and retry/backoff + cache for metadata fetches. Parallel with step 2.
5. Phase 2: Improve crop quality logic by relaxing strict torso keypoint requirements and using segmentation-guided torso fallback before full-image fallback. Depends on step 1.
6. Phase 2: Rebuild representation strategy from one mean vector per pattern to multiple centroids per pattern (for example K-means clusters), then update index build and query-time grouping/reranking. Depends on step 1 and can run in parallel with step 5.
7. Phase 2: Add lightweight sweater-type filtering (cardigan/pullover/etc.) as a pre-filter or rerank feature for nearest-neighbor results. Depends on step 6 if classifier logits are used for reranking, otherwise parallel with step 6.
8. Phase 2: Evaluate retrieval quality on held-out labeled examples and compare old vs new with Recall@K and type-consistency metrics. Depends on steps 5 to 7.
9. Phase 2.5: Improve XAI heatmap faithfulness. Compare Grad-CAM, Grad-CAM++, and Integrated Gradients with the same input and target class, then choose the best method by quantitative faithfulness metrics (deletion/insertion curves and confidence drop when top-saliency regions are masked). Depends on steps 5 to 8.
10. Phase 2.5: Add structure-aware XAI constraints by intersecting saliency with sweater segmentation masks and suppressing background activations. Store both raw and constrained maps for analysis. Depends on step 9.
11. Phase 2.5: Build a small labeled explanation benchmark set (for example 100 images with torso/sleeve/relevant-pattern-region annotations) and set acceptance gates for XAI quality before rollout. Depends on steps 9 to 10.
12. Phase 3: Reduce deployment footprint via model export/quantization and optional XAI deferral (only on-demand). Depends on phase 2 and 2.5 acceptance because quality must remain stable.
13. Phase 3: Choose hosting target from measured constraints (memory, startup, latency). Candidate defaults: Cloud Run for simplest path; Lambda only after artifact size and cold-start constraints are verified.

**Relevant files**
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/main.py — recommendation endpoint flow, timing hooks, metadata fetch retry/cache, input validation, result reranking integration.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/dino_feature_extraction.py — lazy loading and embedding extraction outputs for reuse.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/YOLO_pose_crop.py — robust crop fallback logic and minimum crop quality gates.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/xaiutil.py — avoid duplicate preprocessing; optional deferred XAI generation.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/index_building_files/build_master_vectors.py — multi-centroid vector generation pipeline.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/index_building_files/build_index.py — centroid-aware index structure and metadata mapping.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/requirements.txt — add only minimal new dependencies for clustering/retry/cache and optional quantization stack.
- /Users/kristenvinh/Documents/Github_repos/Knit_Sweater_Recommender/README.md — deployment profile, benchmark results, and retraining/index rebuild instructions.

**Verification**
1. Record baseline metrics across a fixed sample of user images: median latency, P95 latency, crop failure rate, and Recall@5.
2. After phase 1, verify latency reduction target of at least 25% and no increase in failure rate.
3. After phase 2, verify accuracy gain target of at least 10% on Recall@5 and improved sweater-type consistency.
4. Run API-level manual tests for edge images (small, low-light, occluded torso) and ensure graceful fallbacks.
5. Validate rebuilt index integrity: every centroid maps back to a valid pattern and duplicate suppression works in final top-K.
6. Validate XAI faithfulness: top-saliency deletion should reduce target confidence faster than random masking; insertion should recover confidence faster than random baselines.
7. Validate XAI localization quality on the explanation benchmark set with overlap metrics against annotated sweater-relevant regions.
8. Validate deployment artifact and startup constraints for at least one container runtime and one serverless candidate.

**Decisions**
- Priority set to higher recommendation accuracy.
- Retraining is in scope.
- Hosting target is undecided, so plan preserves optionality with container-first deployment guidance and serverless-readiness checks.
- Include: retrieval quality improvements, crop robustness, runtime optimization, deployment sizing.
- Include: retrieval quality improvements, crop robustness, runtime optimization, deployment sizing, and measurable XAI quality improvements.
- Exclude for now: full product redesign, authentication overhaul beyond basic hardening, and large-scale feedback-learning pipelines.

**Further Considerations**
1. Classifier integration strategy recommendation: Option A pre-filter by type before ANN query, Option B query first then rerank by type confidence. Recommend Option B initially to reduce false negatives.
2. XAI strategy recommendation: Option A generate heatmap only when user opens details, Option B keep eager generation. Recommend Option A for lower median latency after phase 2.5 quality gates are passing.
3. Hosting recommendation: default to Cloud Run for first production release, then revisit Lambda only after quantization and package-size profiling.