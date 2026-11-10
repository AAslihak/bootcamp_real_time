% Place analyze_arrivals.m on the MATLAB path or one directory above outputs.
testDir = fileparts(mfilename('fullpath'));
addpath(testDir,fileparts(fileparts(testDir)));
[receiverPath,templatePath,expectedOnsets] = generate_chirp_test(testDir);
template = struct('name','3–15 kHz chirp','wav',templatePath);
opts = struct('BandHz',[3000 15000],'MaxChannelLagSec',0.002,'GuardSec',0.02);
[arrivals,segments] = analyze_arrivals(receiverPath,template,opts);

matched = arrivals(arrivals.Identity == string(template.name),:);
errorSec = abs(matched.TimeSec - expectedOnsets(:,1)');
found = any(errorSec <= 0.002,1);
extras = sum(~any(errorSec <= 0.002,2));
fprintf('Recovered %d/%d expected chirps within 2 ms. Extra matched rows: %d.\n', ...
    nnz(found),numel(found),extras);
