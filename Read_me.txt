python csv_seo_knotknot.py \
  --input products_export.csv \
  --output-dir ./out \
  --limit 10 \
  --model gpt-4o-mini \
  --temperature 0.4





python csv_seo_knotknot.py --input products_export.csv --verbose --output-dir ./out --limit 10 --model gpt-4o-mini 



There you go

Reverse image search (optional):
- Set one of these env vars:
  - BING_VISUAL_SEARCH_KEY=your_key    (or BING_SEARCH_V7_SUBSCRIPTION_KEY)
  - SERPAPI_API_KEY=your_key           (for Google Lens via SerpAPI)
- Then run with --reverse-search to compare your product photos against the web and mine candidate names.

Examples:
  # Visual-first + reverse image search, verbose
  OPENAI_API_KEY=... BING_VISUAL_SEARCH_KEY=... \
  python csv_seo_knotknot.py --input products_export.csv --output-dir ./out \
    --limit 10 --model gpt-4o-mini --visual-first --reverse-search --verbose

Notes:
- The script already sends product image URLs to the model for naming; --reverse-search augments this with external titles from other sites to reduce generic or wrong names.
- Choose provider by setting REVERSE_SEARCH_PROVIDER=bing (default) or serpapi.
