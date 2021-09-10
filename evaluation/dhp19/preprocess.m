tic

% Set recording to pre-process
subj = 1;
sess = 3;
mov = 6;
% Set the paths of code repository folder, data folder and output folder 
rootCodeFolder = '/home/fdipietro/projects/hpe-core'; % root directory of the git repo.
rootDataFolder = '/mnt/0058555E5855540E'; % root directory of the data downloaded from resiliosync.
outFolder = '/home/fdipietro/hpe-data/DVS';

addpath(fullfile(rootCodeFolder));
addpath(fullfile(rootCodeFolder, 'evaluation/dhp19/'));
addpath(fullfile(rootCodeFolder, 'evaluation/dhp19/utils/matlab'));
outFile_string = sprintf('S%d_%d_%d.mat',subj,sess,mov);

if(not(exist(outFile_string, 'file') == 2))
    filterAndSeparateChannels(subj, sess, mov, rootCodeFolder, rootDataFolder, outFolder);
else
    fprintf('File already esists: %s\n', outFile_string);  
end          

toc