from setuptools import setup
from glob import glob

package_name = 'atlaspushup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/rviz',   glob('rviz/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='filip',
    maintainer_email='62928305+ffutera@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ClassicPushup = atlaspushup.ClassicPushup:main',
            'HumanPushup = atlaspushup.HumanPushup:main',
            'AsymmetricPushup = atlaspushup.AsymmetricPushup:main',
            'NoslidePushup = atlaspushup.NoslidePushup:main',
            'asymmetricpushupstand = atlaspushup.asymmetricpushupstand:main',
        ],
    },
)
