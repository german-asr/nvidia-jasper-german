Bootstrap: docker
Registry: nvcr.io
From: nvidia/pytorch:19.09-py3

%runscript

    echo "Nothing to do here."

%files
    requirements.txt /requirements.txt

%post

    apt-get update
    apt-get install -y libsndfile1
    apt-get install -y sox
    rm -rf /var/lib/apt/lists/*

    pip install --disable-pip-version-check -U -r /requirements.txt
