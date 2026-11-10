function [arrivals, segments] = analyze_arrivals(wavPath, templates, opt)
% ANALYZE_ARRIVALS Analyze a saved receiver WAV without using DAQ hardware.
% [A,S] = analyze_arrivals("daq_capture.wav");
% T = struct('name',"m-code",'wav',"measured_transmit.wav");
% [A,S] = analyze_arrivals("daq_capture.wav",T,struct('MaxChannelLagSec',0.012));
% Alternatively supply T.samples (mono) and T.fs. Template sample 1 defines
% the arrival time. Crop transmit lead-in silence before supplying it.
% S{k}.raw preserves all receiver channels on the same sample indices.
% Set opt.Plot=false for batch runs. Requires MATLAB and Signal Processing Toolbox.

if nargin < 1 || isempty(wavPath), wavPath = 'daq_capture.wav'; end
if nargin < 2, templates = []; end
if nargin < 3, opt = struct; end
defaults = struct('BandHz',[],'MaxChannelLagSec',0.020,'NoiseSec',4, ...
    'GuardSec',0.15,'ThresholdMAD',8,'MinScoreDB',5,'Plot',true);
fields = fieldnames(defaults);
for k = 1:numel(fields)
    key = fields{k};
    if ~isfield(opt,key), opt.(key) = defaults.(key); end
end
[x,fs] = audioread(wavPath);
[n,nc] = size(x);
assert(n >= 32 && all(isfinite(x(:))), 'WAV is too short or contains nonfinite samples.');
if isempty(opt.BandHz), opt.BandHz = [min(300,fs/10), 0.45*fs]; end
band = opt.BandHz;
assert(numel(band)==2 && 0 < band(1) && band(1)<band(2) && band(2)<fs/2, ...
    'BandHz must be increasing and strictly inside (0, Nyquist).');
w = min(n,max(32,round(.025*fs)));
hop = max(1,round(w/4));
win = hann(w,'periodic');
z = x - median(x,1);
hits = zeros(0,5); % sample, last sample, channel, template index, score excess dB
for ch = 1:nc
    [p,f,t] = timePower(z(:,ch),fs,w,hop,win);
    use = f>=band(1) & f<=band(2);
    assert(any(use),'No spectrogram bins in BandHz.');
    noise = movmedian(p(use,:),max(3,round(opt.NoiseSec*fs/hop)),2);
    ratio = p(use,:)./max(noise,realmin);
    % Average the strongest bins so a narrow line can survive a wide search.
    ratio = sort(ratio,1,'descend');
    score = 10*log10(max(mean(ratio(1:max(1,ceil(.05*nnz(use))),:),1),realmin));
    threshold = localThreshold(score,hop/fs,opt);
    active = reshape(score > threshold,1,[]);
    starts = find(diff([false active false])==1);
    stops = find(diff([false active false])==-1)-1;
    for j = 1:numel(starts)
        ix = starts(j):stops(j);
        if numel(ix)<3, continue; end
        hits(end+1,:) = [max(1,round(t(ix(1))*fs-w/2)+1), ...
            min(n,round(t(ix(end))*fs+w/2)),ch,0,max(score(ix)-threshold(ix))];
    end
end

names = strings(1,numel(templates));
for k = 1:numel(templates)
    names(k) = string(templates(k).name);
    if isfield(templates,'wav') && ~isempty(templates(k).wav)
        [s,sfs] = audioread(templates(k).wav);
    else
        s = templates(k).samples;
        sfs = templates(k).fs;
    end
    assert(isvector(s),'Transmit template must be mono.');
    s = resample(double(s(:)),fs,sfs);
    s = s - mean(s);
    m = numel(s);
    assert(m>=4 && m<=n && norm(s)>0,'Template must be nonzero and fit inside the WAV.');
    % FFT blocks share input context. Each possible onset belongs to one core.
    step = max(1,round(2*fs));
    context = max(m,round(opt.NoiseSec*fs/2));
    for ch = 1:nc
        score = zeros(n-m+1,1);
        for a = 1:step:numel(score)
            b = min(numel(score),a+step-1);
            lo = max(1,a-context);
            hi = min(n,b+m-1+context);
            y = z(lo:hi,ch);
            [p,f] = timePower(y,fs,w,hop,win);
            psd = median(p,2);
            nf = 2^nextpow2(numel(y)+m-1);
            ff = (0:nf-1)'*fs/nf;
            ff = min(ff,fs-ff);
            weight = interp1(f,psd,ff,'linear','extrap');
            floorPSD = max(realmin,0.01*median(psd(f>=band(1)&f<=band(2))));
            weight = double(ff>=band(1)&ff<=band(2))./max(weight,floorPSD);
            S = fft(s,nf);
            power = sum(abs(S).^2.*weight)/nf;
            assert(power>0,'Template has no usable energy in BandHz.');
            c = ifft(fft(y,nf).*conj(S).*weight,'symmetric')/sqrt(power);
            env = abs(hilbert(c(1:numel(y)-m+1)));
            score(a:b) = 20*log10(max(env(a-lo+1:b-lo+1),realmin));
        end
        threshold = localThreshold(score,1/fs,opt);
        % Suppress adjacent carrier/sidelobe peaks using template autocorrelation.
        ac = abs(hilbert(xcorr(s)));
        width = find(ac(m+1:end)<0.5*ac(m),1);
        if isempty(width), width = m; end
        if numel(score)<3, continue; end
        [~,idx] = findpeaks(score,'MinPeakDistance',min(width,numel(score)-1));
        idx = idx(score(idx)>threshold(idx));
        hits = [hits; idx,idx+m-1,repmat([ch,k],numel(idx),1),score(idx)-threshold(idx)];
    end
end

[rows,segments] = assemble(hits,x,fs,names,band,opt);
arrivals = struct2table(rows(:),'AsArray',true);
disp(arrivals);
if opt.Plot
    figure('Name','Arrival inspection');
    tiledlayout(nc,1);
    for ch = 1:nc
        [p,f,t] = timePower(z(:,ch),fs,w,hop,win);
        nexttile;
        imagesc(t,f/1000,10*log10(max(p,realmin))); axis xy;
        ylim(band/1000); ylabel(sprintf('Ch %d (kHz)',ch));
        hold on;
        for j = 1:height(arrivals)
            xline(arrivals.TimeSec(j),'w');
        end
    end
    xlabel('Receiver time (s)');
end
end

function [p,f,t] = timePower(x,fs,w,hop,win)
[~,f,t,p] = spectrogram(x,win,w-hop,w,fs);
end

function threshold = localThreshold(score,dt,opt)
% Exclude nearby cells and use both flanks. Repeat after masking candidates.
step = max(1,round(.05/dt));
guard = max(1,round(opt.GuardSec/dt));
span = max(guard+1,round(opt.NoiseSec/(2*dt)));
grid = unique([1:step:numel(score),numel(score)]);
base = zeros(size(grid));
excluded = false(size(score));
for pass = 1:2
    for j = 1:numel(grid)
        k = grid(j);
        ix = [max(1,k-span):k-guard, k+guard:min(numel(score),k+span)];
        ix = ix(~excluded(ix));
        if numel(ix)<12
            ref = score(~excluded);
            if isempty(ref), ref = score; end
        else
            ref = score(ix);
        end
        center = median(ref);
        spread = 1.4826*median(abs(ref-center));
        base(j) = center + max(opt.MinScoreDB,opt.ThresholdMAD*spread);
    end
    if numel(grid)==1
        threshold = base(1);
    else
        threshold = reshape(interp1(grid,base,1:numel(score)),size(score));
    end
    excluded = movmax(score>threshold,[guard guard]);
end
end

function [rows,segments] = assemble(hits,x,fs,names,band,opt)
rows = struct('TimeSec',{},'EndSec',{},'Identity',{},'Channels',{}, ...
    'RelativeMs',{},'ScoreExcessDB',{},'SNRdB',{},'Peak',{},'DC',{}, ...
    'FlatTop',{},'NoiseOK',{},'Overlapping',{});
segments = {};
if isempty(hits), return; end
% Keep unknown activity only where it does not overlap a template detection.
known = hits(hits(:,4)>0,:);
unknown = find(hits(:,4)==0);
for j = unknown'
    if any(known(:,1)<=hits(j,2) & known(:,2)>=hits(j,1))
        hits(j,4) = -1;
    end
end
hits = sortrows(hits(hits(:,4)>=0,:),1);
used = false(size(hits,1),1);
lag = round(opt.MaxChannelLagSec*fs);
pad = round(.02*fs);
[n,nc] = size(x);
occupied = false(n,1);
for j = 1:size(hits,1)
    occupied(max(1,hits(j,1)-pad):min(n,hits(j,2)+pad)) = true;
end
[b,a] = butter(4,band/(fs/2),'bandpass');
filtered = filtfilt(b,a,x);
for j = 1:size(hits,1)
    if used(j), continue; end
    eligible = find(~used & hits(:,4)==hits(j,4) & abs(hits(:,1)-hits(j,1))<=lag);
    members = j;
    for ch = setdiff(unique(hits(eligible,3)),hits(j,3))'
        choices = eligible(hits(eligible,3)==ch);
        [~,ii] = min(abs(hits(choices,1)-hits(j,1)));
        members(end+1) = choices(ii);
    end
    used(members) = true;
    start = min(hits(members,1));
    stop = max(hits(members,2));
    lo = max(1,start-pad); hi = min(n,stop+pad);
    raw = x(lo:hi,:);
    taper = ones(size(raw,1),1);
    edge = min([pad,start-lo,hi-stop]);
    if edge > 0
        ramp = (1-cos(pi*(0:edge-1)'/edge))/2;
        taper(1:edge) = ramp;
        taper(end-edge+1:end) = flipud(ramp);
    end
    around = max(1,start-round(opt.NoiseSec*fs/2)):min(n,stop+round(opt.NoiseSec*fs/2));
    bg = around(~occupied(around));
    noiseOK = numel(bg)>=max(round(.1*fs),stop-start+1);
    snrDB = nan(1,nc);
    if noiseOK
        pn = mean(filtered(bg,:).^2,1);
        ps = mean(filtered(start:stop,:).^2,1)-pn;
        valid = ps>0 & pn>0;
        snrDB(valid) = 10*log10(ps(valid)./pn(valid));
    end
    relative = nan(1,nc);
    if hits(j,4)>0
        relative(hits(members,3)) = (hits(members,1)-start)*1000/fs;
        identity = names(hits(j,4));
    else
        identity = "unidentified";
    end
    active = x(start:stop,:);
    flat = false(1,nc);
    if size(active,1)>=3
        same = diff(active,1,1)==0;
        extreme = active(2:end-1,:)==max(active,[],1) | active(2:end-1,:)==min(active,[],1);
        flat = any(same(1:end-1,:) & same(2:end,:) & extreme,1);
    end
    rows(end+1) = struct('TimeSec',(start-1)/fs,'EndSec',stop/fs, ...
        'Identity',identity,'Channels',{hits(members,3)'},'RelativeMs',relative, ...
        'ScoreExcessDB',max(hits(members,5)),'SNRdB',snrDB, ...
        'Peak',max(abs(active),[],1),'DC',mean(active,1),'FlatTop',flat, ...
        'NoiseOK',noiseOK,'Overlapping',false);
    segments{end+1} = struct('firstSample',lo,'lastSample',hi,'fs',fs, ...
        'raw',raw,'tapered',raw.*taper);
end
for j = 1:numel(rows)
    others = setdiff(1:numel(rows),j);
    rows(j).Overlapping = any([rows(others).TimeSec]<rows(j).EndSec & ...
        [rows(others).EndSec]>rows(j).TimeSec);
end
end
