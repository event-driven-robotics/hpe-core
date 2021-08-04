tic

% Set the paths of code repository folder, data folder and output folder 
rootCodeFolder = '/home/fdipietro/projects/hpe-core'; % root directory of the git repo.
rootDataFolder = '/mnt/0058555E5855540E'; % root directory of the data downloaded from resiliosync.
outFolder = '/home/fdipietro/hpe-data/DVS';

addpath(fullfile(rootCodeFolder));
addpath(fullfile(rootCodeFolder, 'evaluation/dhp19/'));
addpath(fullfile(rootCodeFolder, 'evaluation/dhp19/utils/'));

DVSrecFolder = fullfile(rootDataFolder,'DVS_movies/');
numSubjects = 17;
numSessions = 5;

for subj = 1:numSubjects 
    subj_string = sprintf('S%d',subj);
    sessionsPath = fullfile(DVSrecFolder, subj_string);
    
    for sess = 1:numSessions
        sessString = sprintf('session%d',sess);

        movementsPath = fullfile(sessionsPath, sessString);
        
        if     sess == 1, numMovements = 8;
        elseif sess == 2, numMovements = 6;
        elseif sess == 3, numMovements = 6;
        elseif sess == 4, numMovements = 6;
        elseif sess == 5, numMovements = 7;
        end
        
        for mov = 1:numMovements
            outFile_string = sprintf('S%d_%d_%d.mat',subj,sess,mov);

            if (not(exist(outFile_string, 'file') == 2))
                filterAndSeparateChannels(subj, sess, mov, rootCodeFolder, rootDataFolder, outFolder);
            else
                
               fprintf('File already esists: %s\n', outFile_string);  
            end
             
        end 
    end
end


toc