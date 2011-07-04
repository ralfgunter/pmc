arg_list = argv();

for i = 1:nargin
    printf("Parsing simulation number %d\n", i);

    fluence_filename = arg_list{i};
    printf("fluence file: %s\n", fluence_filename);

    fid = fopen(fluence_filename);
    A = fread(fid, 250*70*60, 'float');
    fclose(fid);

    B = reshape(A, [250 70 60]);
    I = figure;
    imagesc(rot90(squeeze(log(abs(B(125,:,:)))), 3), [0 14]);
    colorbar;
    print([fluence_filename, '.png'], I);
end
