python scraper.py \
  --category "https://www.kruidvat.nl/verzorging/haarstylingproducten" \
  --db kruidvat.db 

python scraper.py \
  --category "https://www.kruidvat.nl/verzorging/haarverzorging" \
  --db kruidvat.db 

# Embed everything that was just scraped so the catalogue is searchable.
python embed.py --db kruidvat.db
