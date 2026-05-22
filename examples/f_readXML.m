function settings = f_readXML(path)

metadata = xmlread(path);
nodes = metadata.getElementsByTagName('PVStateValue');

for k = 0:nodes.getLength-1
    node = nodes.item(k);
    key = char(node.getAttribute('key'));

    if strcmp(key, 'framePeriod')
        framePeriod = str2double(char(node.getAttribute('value')));
        frameRate = 1 / framePeriod;
    end

    if strcmp(key, 'rastersPerFrame')
        rastersPerFrame = str2double(char(node.getAttribute('value')));
    end

    if strcmp(key, 'objectiveLens')
        objectiveLens = char(node.getAttribute('value'));
    end

    if strcmp(key, 'activeMode')
        activeMode = char(node.getAttribute('value'));
    end

    if strcmp(key, 'objectiveLensMag')
        objectiveLensMag = str2double(char(node.getAttribute('value')));
    end

    if strcmp(key, 'objectiveLensNA')
        objectiveLensNA = str2double(char(node.getAttribute('value')));
    end

    if strcmp(key, 'opticalZoom')
        opticalZoom = str2double(char(node.getAttribute('value')));
    end

    if strcmp(key, 'micronsPerPixel')
        children = node.getElementsByTagName('IndexedValue');
        
        for j = 0:children.getLength-1
            child = children.item(j);
            index = char(child.getAttribute('index'));
            value = str2double(char(child.getAttribute('value')));

            if strcmp(index, 'XAxis')
                micronsPerPixel_X = value;
            elseif strcmp(index, 'YAxis')
                micronsPerPixel_Y = value;
            elseif strcmp(index, 'ZAxis')
                micronsPerPixel_Z = value;
            end
        end
    end

    if strcmp(key, 'laserPower')
        children = node.getElementsByTagName('IndexedValue');
        
        for j = 0:children.getLength-1
            child = children.item(j);
            index = char(child.getAttribute('index'));
            value = str2double(char(child.getAttribute('value')));

            if strcmp(index, '0')
                laserPower = value;
            end
        end
    end

    if strcmp(key, 'laserWavelength')
        children = node.getElementsByTagName('IndexedValue');
        
        for j = 0:children.getLength-1
            child = children.item(j);
            index = char(child.getAttribute('index'));
            value = str2double(char(child.getAttribute('value')));

            if strcmp(index, '0')
                laserWavelength = value;
            end
        end
    end

    if strcmp(key, 'pmtGain')
        children = node.getElementsByTagName('IndexedValue');
        
        for j = 0:children.getLength-1
            child = children.item(j);
            index = char(child.getAttribute('index'));
            value = str2double(char(child.getAttribute('value')));

            if strcmp(index, '0')
                pmtGain_FarRed = value;
            elseif strcmp(index, '1')
                pmtGain_Red = value;
            elseif strcmp(index, '2')
                pmtGain_Green = value;
            elseif strcmp(index, '3')
                pmtGain_Blue = value;
            end
        end
    end

    if strcmp(key, 'positionCurrent')
        children = node.getElementsByTagName('SubindexedValue');
        child = children.item(0);
        position_Z = str2double(char(child.getAttribute('value')));
        
    end

end

settings = struct;
settings.frameRate = frameRate / rastersPerFrame;
settings.rastersPerFrame = rastersPerFrame;

settings.objective.Lens = objectiveLens;
settings.objective.LensMag = objectiveLensMag;
settings.objective.LensNA = objectiveLensNA;

settings.opticalZoom = opticalZoom;

settings.micronsPerPixel.X = micronsPerPixel_X;
settings.micronsPerPixel.Y = micronsPerPixel_Y;
settings.micronsPerPixel.Z = micronsPerPixel_Z;

settings.laser.Power = laserPower;
settings.laser.Wavelength = laserWavelength;

settings.activeMode = activeMode;

settings.pmtGain.FarRed = pmtGain_FarRed;
settings.pmtGain.Red = pmtGain_Red;
settings.pmtGain.Green = pmtGain_Green;
settings.pmtGain.Blue = pmtGain_Blue;

settings.position_Z = position_Z;

end