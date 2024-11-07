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


from setuptools import setup 
  
setup( 
    name='wordnet_gibberish', 
    version='0.1', 
    description='This word does not exist, word or definition generator.', 
    packages=['wordnet_gibberish'], 
    install_requires=[ 
        'pandas',
        'gibberify',
        'wordfreq',
        'unidecode',
        'stanza',
        'torch',
    ], 
)

# Install the package using the following command:
# pip install -e .