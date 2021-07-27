function [] =saveCams(cam, X, y, timeStamp, pol, startIndex, stopIndex, startTime, stopTime, outFile)

    % Indexes for all 4 cameras    
    idx0 = find(cam == 0);
    idx1 = find(cam == 1);
    idx2 = find(cam == 2);
    idx3 = find(cam == 3);
    % Camera 0
    x0 = X(idx0);
    y0 = y(idx0);
    ts0 = timeStamp(idx0);
    pol0 = pol(idx0);
    % Camera 1
    x1 = X(idx1);
    y1 = y(idx1);
    ts1 = timeStamp(idx1);
    pol1 = pol(idx1);
    % Camera 2
    x2 = X(idx2);
    y2 = y(idx2);
    ts2 = timeStamp(idx2);
    pol2 = pol(idx2);
    % Camera 3
    x3 = X(idx3);
    y3 = y(idx3);
    ts3 = timeStamp(idx3);
    pol3 = pol(idx3);

    %%uotput structure
    out.data.cam0.dvs.x = x0;
    out.data.cam0.dvs.y = y0;
    out.data.cam0.dvs.ts = ts0;
    out.data.cam0.dvs.pol = pol0;

    out.data.cam1.dvs.x = x1;
    out.data.cam1.dvs.y = y1;
    out.data.cam1.dvs.ts = ts1;
    out.data.cam1.dvs.pol = pol1;

    out.data.cam2.dvs.x = x2;
    out.data.cam2.dvs.y = y2;
    out.data.cam2.dvs.ts = ts2;
    out.data.cam2.dvs.pol = pol2;

    out.data.cam3.dvs.x = x3;
    out.data.cam3.dvs.y = y3;
    out.data.cam3.dvs.ts = ts3;
    out.data.cam3.dvs.pol = pol3;

    out.extra.ts = timeStamp;
    out.extra.startIndex = startIndex;
    out.extra.stopIndex = stopIndex;
    out.extra.startTime = startTime;
    out.extra.stopTime = stopTime;


    save(outFile, 'out')
end