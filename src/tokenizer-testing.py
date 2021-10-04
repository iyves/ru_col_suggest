import treetaggerwrapper

res = "Любое предложение, можно ставить любое предложение."

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
treetagger_dir = config['PATHS']['treetagger_dir']

tagger = treetaggerwrapper.TreeTagger(TAGLANG='ru', TAGDIR=str(Path(treetagger_dir)))
tags = tagger.tag_text(res)
tags = [t.split() for t in tags]