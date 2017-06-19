from setuptools import setup

setup(
    name='lace',
    version='0.1.0',
    description='Neural Learning to Rank using Chainer',
    url='https://github.com/rjagerman/lace',
    download_url = 'https://github.com/rjagerman/lace/archive/v0.1.0.tar.gz',
    author='Rolf Jagerman',
    author_email='rjagerman@gmail.com',
    license='MIT',
    packages=['lace',
              'lace.functions',
              'lace.loss',
              'test',
              'test.examples',
              'test.functions',
              'test.loss'],
    install_requires=['numpy>=1.12.0',
                      'chainer>=2.0.0'],
    test_suite='nose.collector',
    tests_require=['nose']
)
