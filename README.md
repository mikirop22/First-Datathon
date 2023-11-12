# UPC Datathon 2023
# Fashion Compatibility Challenge <img src="resources/icon.png" align="right" height=100/>

## Overview

## Explicació codi

## Preprocessing product_data

Per realitzar el preprocessament inicial de product_data, es van descartar totes les dades considerades prescindibles i es va suprimir completament la fila corresponent.



## Outfit_data

Durant el procés de preprocessament de les dades d'outfit_data, es van excloure els elements dels conjunts de vestuari que havien sigut eliminats durant la manipulació prèvia de les dades de producte.

Per dur a terme aquest procés, es va obrir el fitxer .csv de product_data i es van conservar les seves dades. Posteriorment, es va accedir al conjunt de dades d'outfit_data. Seguidament, es va identificar els productes que no pertanyien a cap conjunt de vestuari i es van eliminar aquestes peces de roba.


## Repreprocessing product_data

Finalment, per optimitzar la classificació dels diferents tipus de peces de vestuari, es va realitzar una modificació a la columna "des_product_category", unificant tots els tipus de calçat sota una única categoria anomenada 'shoes'.


## new_outfit.py

Crea la nova base de dades de l'outfit preprocessada.
Aquest script pren dades d'imatges i dades d'ataviaments des de dos arxius CSV separats. Extreu les rutes de les imatges, processant-les per emmagatzemar-les. Després, recull les dades dels atuendos i les relaciona amb les imatges disponibles. Finalment, crea un nou arxiu CSV amb aquestes dades preparades per a un ús posterior.

## faster.py

La secció inicial del codi carrega dades d'arxius CSV i les prepara per a l'anàlisi. Les columnes categòriques es converteixen en valors numèrics per calcular similituds. La funció `calculate_similarity_based_on_metadata` mesura la similitud entre imatges i els seus metadades, incloent la similitud del conjunt d'indumentària.

A continuació, el codi ordena les similituds i analitza la semblança entre les imatges. Després, es genera un procés per a crear vestits amb diferents peces de roba. Aquest algoritme sembla garantir la diversitat de peces en el vestit, evitant repeticions i prioritzant la inclusió de diferents tipus de roba.

Després d'això, el codi organitza el vestit seguint una llista de categories de roba. La selecció i organització de les peces es basa en aquest ordre específic.

El codi visualitza aquests vestits en una interfície gràfica, permetent que els usuaris interactuïn amb la recomanació. Així, l'usuari pot treure elements del vestit i el programa actualitzarà la recomanació en consequència. A més, ofereix opcions per copiar i enganxar el vestit recomanat.

En resum, aquest codi està dissenyat per generar i recomanar conjunts de roba basats en similituds entre les peces i ofereix una interfície per interactuar amb aquestes recomanacions.

NOTA: Hi ha una secció, la funció calculate_similarity_based_on_metadata() que està parcialment inacabada. Els pesos dels outfits no són del tot correctes i per tant recomanem no tenir-los en compte. Aleshores, escollirà els outfits basant-se en peçes de roba similars i no tindrà en compte els outfits existents.
## outfit_embed

Calcula les mitjanes dels embeddings de cada outfit.
