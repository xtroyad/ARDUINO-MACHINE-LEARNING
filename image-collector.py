from jmd_imagescraper.core import *
from pathlib import Path

# root = Path().cwd()/"imagenes"

# duckduckgo_search(root, "Tuerca", "steel nut", max_results=50)
# duckduckgo_search(root, "Tornillo", "screw", max_results=50)
# duckduckgo_search(root, "Arandela", "Stainless Steel Flat Washer", max_results=50)
# duckduckgo_search(root, "Mariposa", "wing nut", max_results=50)


root = Path().cwd()/"/p"
duckduckgo_search(root, "prueba","lil steel nut", max_results=5)
duckduckgo_search(root,"prueba", "lil screw", max_results=5)
duckduckgo_search(root, "prueba","Stainless Steel ", max_results=5)
duckduckgo_search(root,  "prueba","lil wing nut", max_results=5)

