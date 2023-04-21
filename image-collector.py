from jmd_imagescraper.core import *
from pathlib import Path

# root = Path().cwd()/"imagenes"

# duckduckgo_search(root, "Tuerca", "steel nut", max_results=50)
# duckduckgo_search(root, "Tornillo", "screw", max_results=50)
# duckduckgo_search(root, "Arandela", "Stainless Steel Flat Washer", max_results=50)
# duckduckgo_search(root, "Mariposa", "wing nut", max_results=50)


root = Path().cwd()/"p"
duckduckgo_search(root, "Tuerca","one steel nut", max_results=200)
duckduckgo_search(root,"Tornillo", "one screw", max_results=200)
duckduckgo_search(root, "Arandela","one Stainless Steel ", max_results=200)
duckduckgo_search(root,  "Mariposa","one wing nut", max_results=200)

