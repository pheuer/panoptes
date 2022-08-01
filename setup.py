import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='panoptes',
    version='0.0.1',
    author='Peter Heuer',
    author_email='pheu@lle.rochester.edu',
    description='Analysis code for HEDP imaging diagnostics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pheuer/panoptes',
    project_urls = {
        "Bug Tracker": "https://github.com/pheuer/panoptes/issues"
    },
    license='MIT',
    packages=['panoptes'],
    install_requires=['numpy', 'h5py', 'scipy', 'matplotlib', 'astropy', 'CR39py'],
)