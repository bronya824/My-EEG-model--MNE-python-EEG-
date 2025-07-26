from setuptools import setup, find_packages

setup(
    name='eeg_proc',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.0',
        'mne>=1.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pydantic>=2.0',  # 升级到v2
        'alibabacloud_oss20190517',  # 修正阿里云包名
        'pyedflib',
        'websockets',
        'colorlog',  # 文档17使用
        'tqdm',
        'werkzeug>=2.0.0',  # 新增依赖
        'psutil',  # 新增依赖
        'python-dotenv'  # 新增依赖
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov'
        ]
    },
    entry_points={
        'console_scripts': [
            # 指向文档17的批处理函数
            'eeg_batch = eeg_proc.processing.batch_processor:run_batch_analysis',
            # 可添加实时处理入口
            'eeg_realtime = eeg_proc.examples.realtime_processing:start_processing'
        ],
    },
    include_package_data=True,
    package_data={
        # 包含必要的配置文件
        'eeg_proc': ['config/*.yaml']
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)