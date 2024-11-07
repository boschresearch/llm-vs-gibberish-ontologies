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


from wordnet_gibberish.database import RDFDatabase
import argparse

def main(args):
    save_dir = args.save_dir
    rdfdb = RDFDatabase(
                        kg_name=f"WordNet-{args.which}",
                        init_kg=False,
                        add_prefixes=False
    )
    rdfdb.export(
        save_dir,
        args.which
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the RDF database to a directory.")
    parser.add_argument("--which", type=str, help="Which WordNet to use (sweets, football, music)", required=True)
    parser.add_argument("--save_dir", type=str, help="Directory to save the RDF database.", required=True)
    args = parser.parse_args()
    main(args)