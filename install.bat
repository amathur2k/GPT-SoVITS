call conda install -c conda-forge gcc -y
call echo Done 1
call conda install -c conda-forge gxx -y
call echo Done 2
call conda install ffmpeg cmake -y
call echo Done 3
call conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
call echo Done 4
call pip install -r requirements.txt
call echo Done 5