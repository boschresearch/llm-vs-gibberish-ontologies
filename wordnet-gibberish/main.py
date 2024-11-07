# Experiment resources related to the paper "Do LLMs Really Adapt to Domains? An Ontology Learning Perspective"
# (ISWC 2024)
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from wordnet_gibberish.translator import PureGibberishTranslator, GPT2Translator
from wordnet_gibberish.database import RDFDatabase
from wordnet_gibberish.connections import Pickaxe
from wordnet_gibberish.connections.connect_concepts import find_related_concepts, \
    create_connectivity_table, update_kg
from wordnet_gibberish.utils.utils import get_top_words
from wordnet_gibberish.connections.replace import Replacer


from stanza import Pipeline
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import argparse
import logging


def main(args):
    pipeline = Pipeline(lang="en", processors="tokenize,mwt,pos,lemma", download_method=None)
    if args.translator == "puregibberish":
        translator = PureGibberishTranslator(
            seed = args.seed
        )
    elif args.translator == "gpt2":
        translator = GPT2Translator(
            temperature=0.5,
        )
    else:
        raise ValueError(f"Translator {args.translator} not recognized.")

    rdfdb = RDFDatabase(
        kg_name=f"WordNet-{args.which}",
        init_kg=False,
        add_prefixes=True)
    if args.which == "sweets":
        root_concepts = ['<https://en-word.net/id/oewn-07612255-n>',
                        '<https://en-word.net/id/oewn-07625449-n>',
                        '<https://en-word.net/id/oewn-05724524-n>',
                        '<https://en-word.net/id/oewn-05001591-n>',
                        '<https://en-word.net/id/oewn-07875260-n>',
                        '<https://en-word.net/id/oewn-02199916-v>'
                        ]
    elif args.which == "football":
        root_concepts = ['<https://en-word.net/id/oewn-00469555-n>', # football
                        '<https://en-word.net/id/oewn-03383611-n>', # football
                        '<https://en-word.net/id/oewn-08225481-n>', # team, squad (will include offense and defense)
                        '<https://en-word.net/id/oewn-01634178-a>', # defensive (adj)
                        '<https://en-word.net/id/oewn-01633007-a>', # offensive (adj)
                        '<https://en-word.net/id/oewn-00982124-n>' # offensive (noun)
                        ]
    elif args.which == "music":
        # root_concepts = [
        #     '<https://en-word.net/id/oewn-07034009-n>', # music
        #     '<https://en-word.net/id/oewn-05726503-n>', # music
        #     '<https://en-word.net/id/oewn-00544270-n>', # music
        #     '<https://en-word.net/id/oewn-05726882-n>', # music
        # ]
        root_concepts = [
            '<https://en-word.net/id/oewn-03806455-n>', # musical instrument
        ]
    else:
        args.which = "general"
        root_concepts = [
            '<https://en-word.net/id/oewn-07571428-n>',
            '<https://en-word.net/id/oewn-00021445-n>',
        ]
    
    title = f"{args.translator}_{args.which}"
    log_file = Path(__file__).parent / "logs"
    log_file.mkdir(parents=True, exist_ok=True)
    log_file = log_file / f"{title}.log"

    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    # Find out what those root concepts are
    query = f"""
    SELECT DISTINCT ?C ?P ?D (GROUP_CONCAT(DISTINCT ?G; SEPARATOR=";") AS ?F)
    WHERE {{
        VALUES ?C {{ {" ".join(root_concepts)} }}
        ?C rdf:type ontolex:LexicalConcept .
        ?C gwn:partOfSpeech ?P .
        ?C gwn:definition ?D .
        ?C sct:writtenForm ?G .
        FILTER(LANGMATCHES(LANG(?G), "en"))
    }} GROUP BY ?C ?P ?D
    """
    df = rdfdb.issue_query(query, to_table=True)
    print(f"[INFO] Translator : {args.translator}, which: {args.which}")
    print("[INFO] Here is the information about the root concepts. The first 5 rows are:")
    print(df.head())

    path_connected = Path(__file__).parent / "data" / "fake"
    path_connected.mkdir(parents=True, exist_ok=True)
     # We have an initial banlist from ChatGPT
    if args.which == "sweets":
        list_related_words_gpt = "sweet, cake, candy, chocolate, sugar, cookie, dessert, ice cream, pie, pastry, brownie, fudge, caramel, toffee, truffle, macaron, cupcake, donut, pudding, sorbet, sherbet, parfait, mousse, cheesecake, baklava, halva, nougat, marzipan, gumdrop, jellybean, licorice, marshmallow"
        list_related_words_gpt = list_related_words_gpt.split(", ")
    elif args.which == "football":
        list_related_words_gpt = "football, foot ball, offense, defense, defence, offensive, defensive, goalkeeper, goal, attack, touchdown, quarterback, endzone, helmet, tackle, field goal, referee, kickoff, interception, huddle, punt, snap, blitz, fumble, penalty, overtime, sideline, cheerleader, running back, wide receiver, offensive line, defensive line, safety, cornerback, punter, kicker, touchdown dance, instant replay, challenge flag, red zone, two-point conversion, flea flicker, screen pass, draw play, slant route, fade route, post route, curl route, bubble screen, shovel pass, option play, wildcat formation, shotgun formation, spread offense, hurry-up offense, no-huddle offense, nickel defense, dime defense, prevent defense, blitz package, zone coverage, man-to-man coverage, bump-and-run coverage, press coverage, read option, play action, bootleg, quarterback sneak, power run, counter run, sweep, trap, zone blocking, power blocking, pulling guard, pancake block, chop block, cut block, pass rush, bull rush, spin move, swim move, rip move, stunting, zone blitz, cover 2, cover 3, cover 4, cover 6, dime package, nickel package, 3-4 defense, 4-3 defense, 46 defense, Tampa 2 defense, West Coast offense, Air Coryell offense, run and shoot offense, spread option offense, triple option offense, wishbone offense, flexbone offense, pistol formation, read option pass, flea flicker pass, halfback pass, trick play, onside kick, squib kick, touchback, fair catch, kickoff return, punt return, kick coverage, punt coverage, field position, time of possession, turnover margin, redshirt, walk-on, scholarship, recruiting, combine, draft, free agency, salary cap, franchise tag, Pro Bowl, All-Pro, Hall of Fame, Super Bowl, championship, rivalry, tradition, tailgate, fan base, cheer, chant, fight song, mascot"
        list_related_words_gpt = list_related_words_gpt.split(", ")
    elif args.which == "music":
        list_related_words_gpt = "acoustic guitar, bass guitar, electric guitar, drum kit, drum machine, flute, clarinet, saxophone, trumpet, trombone, violin, viola, cello, double bass, piano, keyboard, organ, synthesizer, banjo, mandolin, harp, autoharp, lute, ukulele, xylophone, marimba, vibraphone, conga, bongo, djembe, timpani, glockenspiel, accordion, harmonica, bagpipes, tuba, French horn, oboe, bassoon, sarrusophone, ocarina, sitar, koto, erhu, sheng, ney, duduk, balalaika, didgeridoo, hurdy gurdy, pan flute, slide whistle, melodica, concertina, harmonium, jaw harp, thumb piano, steel drum, tambourine, castanets, triangles, cymbals, bells, chimes, xylophone, vibraphone, marimba, celesta, harpsichord, carillon, jew's harp, mellotron, stylophone, ondes martenot, theremin, kalimba, maramba, guiro, conga, bongos, djembe, gong, temple blocks, surdo, conch shell, ocarina, bamboo flute, nose flute, jaw harp, mouth harp, jaw harp, didgeridoo, jaw harp, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute, nose flute, bamboo flute"
        list_related_words_gpt = list_related_words_gpt.split(", ")
    else:
        list_related_words_gpt = ['cup', 'cafe', 'tea', 'coffee', 'wine', 'food', 'restaurant', 'menu', 'bar', 'eat', 'dinner', 'lunch', 'breakfast', 'snack', 'meal', 'drink', 'juice', 'beer', 'cocktail', 'soda', 'water', 'liquor', 'alcohol', 'bakery', 'grocery', 'market', 'produce', 'fruit', 'vegetable', 'meat', 'seafood', 'cheese', 'bread', 'cake', 'dessert', 'chocolate', 'ice cream', 'sauce', 'spice', 'salt', 'pepper', 'oil', 'butter', 'honey', 'sugar', 'flour', 'rice', 'pasta', 'soup', 'stew', 'grill', 'fry', 'bake', 'cook', 'recipe', 'taste', 'delicious', 'yummy', 'hungry', 'thirsty']
    
    top_words = get_top_words(k=2000, banlist=list_related_words_gpt)

    if not (path_connected / f"{args.which}_connected_concepts.csv").exists():
        pickaxe = Pickaxe(rdfdb, root_concepts)

        explored_concepts = pickaxe.mine(max_depth=5, domain_topic=(args.which in {"football", "music"}))
        explored_concepts_ = set(explored_concepts.keys())
        # Convert the dictionary to a DataFrame

        concepts_df = pd.DataFrame(explored_concepts).T
        print("[INFO] Explored concepts:")
        print(concepts_df.head())

        # Connected concepts to each concept in concepts_df
        connected_series = []
        for definition in tqdm(concepts_df["definition"]):
            connected_series.append(find_related_concepts(definition, pipeline, rdfdb, top_words))
        
        # Add the connected concepts to the DataFrame
        concepts_df["connected"] = connected_series

        concepts_df.to_csv(path_connected / f"{args.which}_connected_concepts.csv", sep="\t")
    else:
        concepts_df = pd.read_csv(path_connected / f"{args.which}_connected_concepts.csv", sep="\t", index_col=0)
        concepts_df["forms"] = concepts_df["forms"].apply(eval)
        explored_concepts_ = set(concepts_df.index)

    print("[INFO] Connected concepts:")
    print(concepts_df.head())

    # We should in fact make a modification table.
    # This table will have the following columns:
    # - concept
    # - definition
    # - token
    # - lemma
    # - pos
    # - related
    if not (path_connected / f"{args.which}_connectivity_table.csv").exists():
        connectivity_df = create_connectivity_table(concepts_df)
        connectivity_df.to_csv(path_connected / f"{args.which}_connectivity_table.csv", sep="\t")
    else:
        # Be cautious: the words "null" and "void" are not to be considered NaN.
        connectivity_df = pd.read_csv(path_connected / f"{args.which}_connectivity_table.csv", sep="\t", index_col=0,
                                      converters={"token" : str, "lemma": str}
                                      )
        # Eval the related column as a list
        connectivity_df["related"] = connectivity_df["related"].apply(eval)

    print("[INFO] Connectivity table:")
    print(connectivity_df.head())

    # assert that all dependencies are internal
    all_internal = connectivity_df["related"].apply(lambda x: all([c in explored_concepts_ for c in x]))
    all_internal = all_internal.all()
    assert all_internal, "Not all dependencies are internal."

    # Now, we can add the connections to the KG.
    update_kg(connectivity_df, rdfdb, verbose=True)

    # First step of replacement.
    replacer = Replacer(translator=translator, rdfdb=rdfdb, concepts_df=concepts_df)
    all_processed_concepts = replacer.first_step()

    def remaining_concepts(explored_concepts):
        query = """SELECT ?C WHERE {
            ?C rdf:type ontolex:LexicalConcept .
            ?C sct:definition ?D .
            ?C sct:writtenForm ?F .
            FILTER(LANGMATCHES(LANG(?F), "fr") && LANGMATCHES(LANG(?D), "fr"))
        }
        """
        fully_processed_concepts = rdfdb.issue_query(query, to_table=True)
        fully_processed_concepts = set(fully_processed_concepts["?C"])
        return explored_concepts - fully_processed_concepts
    
    remaining_concepts_ = remaining_concepts(explored_concepts_)
    # Log remaining concepts after step 1
    logging.info(f"Fully processed concepts after step 1: {explored_concepts_ - remaining_concepts_}\n")
    logging.info(f"Remaining concepts after step 1: {remaining_concepts_}\n")


    len_before = 0
    len_after = len(all_processed_concepts.intersection(explored_concepts_))
    print(f"[INFO] First step completed. {len_after} concepts processed.")
    step = 2

    while len(remaining_concepts_) > 0:
        try:
            print(f"[INFO] Step {step}:")
            if len_after > len_before:
                len_before = len_after
                new_processed_concepts = replacer.next_step(
                    all_processed_concepts, connectivity_df, pipeline
                )
                all_processed_concepts = all_processed_concepts.union(
                    new_processed_concepts.intersection(explored_concepts_)
                )
                len_after = len(all_processed_concepts.intersection(explored_concepts_))
                step += 1
                print(f"[INFO] Number of processed concepts: {len_after - len_before} out of {len(explored_concepts_)}")
                print(f"[INFO] Cumulated number of processed concepts: {len_after} out of {len(explored_concepts_)}")
            else:
                # We will dish out gibberish names (without definitions) until we are able to process some definitions again.
                print("[INFO] Giving gibberish name to a random concept (without giving it a gibberish definition)")
                c = replacer.name_random(remaining_concepts=explored_concepts_ - all_processed_concepts)
                all_processed_concepts.add(c)
                len_after = len(all_processed_concepts.intersection(explored_concepts_))
        except Exception as e:
            print(f"[ERROR] {e}")
            break

        remaining_concepts_ = remaining_concepts(explored_concepts_)
        # Log remaining concepts after step i
        logging.info(f"Remaining concepts after step {step}: {remaining_concepts_}\n")

    print(f"[INFO] Final number of processed concepts: {len_after} out of {len(explored_concepts_)}")
    print("Unprocessed concepts:")
    print(explored_concepts_ - all_processed_concepts)

    # Export the KG
    print(f"[INFO] Exporting KG to {path_connected}.")
    rdfdb.export(
        path_connected,
        title
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--translator", type=str, default="puregibberish", help="Translator to use (puregibberish or gpt2).")
    parser.add_argument("--which", type=str, default="sweets", help="Which concepts to use (sweets, football or music).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator.")
    args = parser.parse_args()
    main(args)
