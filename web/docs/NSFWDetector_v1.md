# NSFWDetector : NSFW PG Detector

This node will define if an image is PG safe.

## Parameters

- **image**: Input image to process
- **profile**: Define for which age range the PG score should define if an image is PG safe (0.0 - 1.0)

```
{
    "Strict (0–6)":       {"nsfw_sum": 0.02, "nsfw_peak": 0.015, "violence": 0.15, "horror": 0.30},
    "Moderate (7–9)":     {"nsfw_sum": 0.03, "nsfw_peak": 0.020, "violence": 0.22, "horror": 0.35},
    "Older kids (10–12)": {"nsfw_sum": 0.05, "nsfw_peak": 0.030, "violence": 0.25, "horror": 0.40},
}
```