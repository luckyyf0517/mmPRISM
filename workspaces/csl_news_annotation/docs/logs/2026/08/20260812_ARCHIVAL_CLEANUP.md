# CSL-News Archival Cleanup

Status: recorded
Owner: CSL-News annotation lane
Recorded at: 2026-08-12 UTC

## Decision And Boundary

The revision data-rebuild priority has moved to CSL-Daily and the new real semantic CSL collection.
CSL-News source acquisition and annotation are retired because their source-download and visual-pose
annotation overhead is disproportionate to their role in the revision. This is not a data-quality
judgment on the completed outputs.

The retained CSL-News artifacts are optional checkpoint-side visual-pose evidence only. They are not an
active source, a resumable annotation run, a final Parquet delivery, or evidence for the required real-radar
generalization experiments.

## Pre-cleanup Inventory

The final inventory immediately before removal recorded:

| Path | Purpose | Size |
|---|---|---:|
| `/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121` | official RGB ZIPs, partial transfers, aria2 state, and labels | 335 GB |
| `/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_annotation` | extracted-video annotation cache | 32 GB |
| `/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_source_integrity` | source-integrity cache | 12 KB |
| `/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2` | live source-consumption registry | 633 KB |
| `/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_manifest_v2` | ZIP-dependent source manifest | 226 MB |
| `/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1` | historical source-integrity reports and registry | 409 KB |
| `/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_manifest_v1` | historical ZIP-dependent source manifest | 27 MB |
| `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1` | source-only audit trial | 39 KB |
| `/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1` | retained annotated outputs and provenance | 13 GB |
| `/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/pose_manifest_v1` | retained frozen pose-manifest snapshots | 214 MB |
| `/mnt/gfs/yanyifan/mmPRISM/splits/csl_news` | retained partial split evidence | 401 KB |

Immediately before cleanup, `/mnt/gfs` reported 2.8 TiB available. The retained output root contained
25,630 NPZ files and 25,630 JSON sidecars, with zero unpaired files, plus 11 archive markers and 52 run
records.

## Removal Set

The following source-only paths are removed as a single irreversible cleanup set:

```text
/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121
/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_annotation
/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_source_integrity
/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2
/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_manifest_v2
/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1
/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v1
/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_manifest_v1
```

No artifact under `pose_annotation/rtmw3d_l_794dbc78_v1`, `pose_manifest_v1`, or `splits/csl_news`
is in the removal set. `/home/yanyifan/upload/20260812/archive_002.zip` is also outside this operation.

## Operational Shutdown

Before deletion, the user-systemd source-integrity and annotation-status timers are stopped and disabled.
The scheduler control file remains `paused`; after the source registry is removed it cannot be resumed.

## Post-cleanup Verification

All listed source-only paths were absent after cleanup. Three incomplete hidden pose-manifest staging
directories were also removed; they were not published snapshots. The retained annotation root still contains
25,630 NPZ files paired one-to-one with 25,630 JSON sidecars, 11 archive markers, and 52 run records, with no
unpaired NPZ or sidecar. The retained pose output, frozen pose-manifest root, and split root occupy about
13 GB, 158 MB, and 401 KB respectively.

`mmprism-csl-news-integrity-scan.timer` and `mmprism-csl-news-annotation-status.timer` have no loaded or
scheduled instances, and no CSL-News download/annotation process is running. The scheduler control file
remains `paused`. `/mnt/gfs` had 3,072,164,290,560 bytes available after cleanup, approximately 2.79 TiB.
