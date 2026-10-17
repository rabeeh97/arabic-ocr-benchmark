#!/usr/bin/env bash
# Fetch the lexicon sources (checklist 1.7) and the modern-text sample (1.8) to the SSD.
# Every source lands in its own directory under $DATA and is treated as untrusted data:
# nothing from it is executed; it is only parsed by scripts/lexicon/build.py under `python -I`.
#
#   bash scripts/lexicon/fetch.sh            # all sources
#   bash scripts/lexicon/fetch.sh wikidata   # one source
#
# Licences are recorded in results/lexicon/LICENCES.md.
set -euo pipefail
DATA="${LEXICON_DATA:-/Volumes/PortableSSD/data}"
LEX="$DATA/lexicon"
GET=(curl -sSL --retry 5 -C -)
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(ayaspell aspell pecina anetac geonames wikidata arwiki)
want() { [[ " ${ARGS[*]} " == *" $1 "* ]]; }

if want ayaspell; then   # Ayaspell Hunspell dictionary, as shipped by LibreOffice (GPL/LGPL/MPL tri-licence)
  mkdir -p "$LEX/ayaspell"
  U=https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ar
  "${GET[@]}" -o "$LEX/ayaspell/ar.dic" "$U/ar.dic"
  "${GET[@]}" -o "$LEX/ayaspell/ar.aff" "$U/ar.aff"
fi

if want aspell; then     # GNU aspell6-ar 1.2-0 (Buckwalter-derived, GPL-2.0-or-later)
  mkdir -p "$LEX/aspell_ar"
  "${GET[@]}" -o "$LEX/aspell_ar/aspell6-ar-1.2-0.tar.bz2" https://ftp.gnu.org/gnu/aspell/dict/ar/aspell6-ar-1.2-0.tar.bz2
  tar -xjf "$LEX/aspell_ar/aspell6-ar-1.2-0.tar.bz2" -C "$LEX/aspell_ar"   # data only; never run its configure
fi

if want pecina; then     # Attia, Pecina, Samih, Shaalan, van Genabith (LREC 2012): 9M inflected forms
  mkdir -p "$LEX/pecina"
  "${GET[@]}" -o "$LEX/pecina/Arabic-Wordlist-1.6.zip" https://sourceforge.net/projects/arabic-wordlist/files/Arabic-Wordlist-1.6.zip/download
fi

if want anetac; then     # ANETAC (Hadj Ameur et al. 2019), transliteration release (no class file, no licence file)
  mkdir -p "$LEX/anetac"
  U="https://raw.githubusercontent.com/HadjAmeur/ANETAC-Dataset/main"
  "${GET[@]}" -o "$LEX/anetac/README.md" "$U/README.md"
  for s in train dev test; do for l in ar en; do
    "${GET[@]}" -o "$LEX/anetac/$s.$l" "$U/EN-AR%20transliteration%20dataset/$s.$l"
  done; done
fi

if want geonames; then   # GeoNames alternate names (CC BY 4.0); build.py streams the zip and keeps isolanguage=ar
  mkdir -p "$LEX/geonames"
  "${GET[@]}" -o "$LEX/geonames/alternateNamesV2.zip" https://download.geonames.org/export/dump/alternateNamesV2.zip
fi

if want wikidata; then   # Wikidata (CC0) via the QLever SPARQL endpoint: Arabic labels and aliases by class, paged
  "${PY:-python3}" -I "$(dirname "$0")/fetch_wikidata.py" --out "$LEX/wikidata"
fi

if want arwiki; then     # Arabic Wikipedia 2023-11-01 (CC BY-SA 4.0 / GFDL), one of seven HF parquet shards
  mkdir -p "$DATA/text/arwiki"
  "${GET[@]}" -o "$DATA/text/arwiki/train-00002-of-00007.parquet" \
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ar/train-00002-of-00007.parquet"
fi
