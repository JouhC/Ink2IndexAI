# Candidate Pair Layout Notes

This note documents the column-window change in `app/pipeline/candidates.py`.
It exists so the behavior can be tuned or backed out without rediscovering the
same page-layout assumptions.

## Problem

The pipeline used inferred `column_id` as the only test for
`same_column_window`. That was too broad on pages with narrow newspaper gutters:
the interval merge step could combine two visual columns into one inferred
layout column, then candidate generation would label a visibly right-hand block
as a same-column candidate.

## Current Behavior

Column inference still uses a merge gap, but the gap is conservative and capped:

- `COLUMN_MERGE_GAP_PAGE_WIDTH_RATIO`
- `COLUMN_MERGE_GAP_MEDIAN_WIDTH_RATIO`
- `COLUMN_MERGE_GAP_MAX_PAGE_WIDTH_RATIO`

Candidate generation now treats `column_id` as a coarse layout signal. A pair
with the same inferred `column_id` is only a `same_column_window` candidate if
it also passes the stricter reading-lane test:

- enough horizontal overlap: `SAME_COLUMN_MIN_X_OVERLAP_RATIO`
- or close enough center x distance:
  `SAME_COLUMN_MAX_CENTER_DX_MEDIAN_WIDTH_RATIO`

If two blocks share an inferred column but fail the same-column window test,
they can still be considered:

- `adjacent_column` when the horizontal gap is small enough
- `cross_column_headline_continuation` when the class mix is plausible for a
  wider continuation

Candidate generation also treats an intervening headline as an article
boundary, but only for headline detections that are visually large enough to
act as separators. Before assigning any source bucket,
`build_page_candidate_pairs()` skips a pair when all of these are true:

- neither candidate block is itself a `Section-header` or `Title`
- a `Title` sits vertically between the two blocks, or a `Section-header` sits
  between them and passes the boundary-size test:
  - height is at least `SECTION_HEADER_BOUNDARY_MIN_HEIGHT_PAGE_RATIO` of the
    page height, or
  - height is at least `SECTION_HEADER_BOUNDARY_MIN_WIDE_HEIGHT_PAGE_RATIO` of
    page height and width is at least
    `SECTION_HEADER_BOUNDARY_MIN_WIDTH_PAGE_RATIO` of page width
- that intervening headline horizontally overlaps both candidate blocks by at
  least `HEADLINE_BOUNDARY_MIN_X_OVERLAP_RATIO`

This prevents vertically adjacent same-column text from crossing into the next
article when a display headline separates the two stories, while allowing small
YOLO `Section-header` detections to remain eligible as article content. Those
bridge pairs are not sent to the pairwise model, so they cannot later merge
clusters through union-find.

## Backtracking

To restore the old behavior, revert the constants and helper functions added in
`app/pipeline/candidates.py`, then change the `column_delta == 0` branch in
`build_page_candidate_pairs()` back to always selecting
`same_column_window`. Also remove the intervening-headline boundary check if
candidate generation should again allow text-to-text pairs across headlines.

For gentler tuning, adjust the constants above instead of removing the logic.
Increasing the merge gap or lowering the same-column thresholds will make the
pipeline behave more like the old version. Lowering the merge gap or raising the
same-column thresholds will make candidate windows more visually strict.
Lowering `HEADLINE_BOUNDARY_MIN_X_OVERLAP_RATIO` makes headline boundaries block
more pairs; raising it makes the guard more permissive.
Lowering the section-header boundary size ratios makes smaller `Section-header`
detections act as article boundaries; raising them limits the boundary guard to
larger display-style headers.
