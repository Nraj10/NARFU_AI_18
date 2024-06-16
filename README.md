# NARFU_AI_18
This is a tool for projecting satellite crops to the sentinel-2 layouts  
## how to setup
`
pip3 install -r module/requirements.txt
`

Set paths to your data in the config.json.
The default path is ./data/. Layouts are in the layouts dir (./data/layouts/).

Before start you need to cache layouts features. You just need to invoke the module without args.

```
python3 -m module
```

The program will cache all the layouts in the destination folder.

## how to use
```
python3 -m module --crop_name <crop path> --layout_name <specific layout>
```

If you will not provide --layout_name, the program will select the layout automatically, using the best match by affine transformation factor of projection matrix.

The result of processing is stored in the coords.csv file.

## export format

The projected coords are stored in these fields: ul, ur, br, bl.
Also the export contains crs, crop_name, layout_name and fix_info fields. Fix info contains information about bad pixels in a crop following this format.

```
x px; y px; channel index; the old value; the new value
x px; y px; channel index; the old value; the new value
...
```
