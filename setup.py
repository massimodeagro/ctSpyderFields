from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ctSpyderFields',
    url='https://github.com/massimodeagro/ctSpyderFields',
    author='Massimo De Agro',
    author_email='massimo.deagro@gmail.com',
    # Needed for dependencies
    install_requires=['numpy', 'opencv-python', 'trimesh', 'tqdm', 'pandas',
                      'scipy', 'networkx', 'pytables', 'matplotlib'],
    # *strongly* suggested for sharing
    version='0.3',
    # The license can be anything you like
    license='GNU',
    description='Python package that project retinas through lenses as segmented from micro CT scans (Amira, Dragonfly) of spiders.',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
