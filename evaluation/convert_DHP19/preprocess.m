tic

% Set recording to pre-process
subj = 1;
sess = 1;
mov = 5;
% Set the paths of code repository folder, data folder and output folder 
rootCodeFolder = '/home/fdipietro/projects/hpe-core'; % root directory of the git repo.
rootDataFolder = '/mnt/0058555E5855540E'; % root directory of the data downloaded from resiliosync.
outFolder = '/home/fdipietro/hpe-data/DVS';

filterAndSeparateChannels(subj, sess, mov, rootCodeFolder, rootDataFolder, outFolder);

toc