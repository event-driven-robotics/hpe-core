In one terminal run yarpdataplayer, and load the location of the data (folder of sample folders) into the dataplayer.

In one termminal run yardatasetver.
In a third terminal run `yarp-example --example_flag true --example_parameter 0.01`
In a fourth terminal connect the dataport from the relevant datafile in the dataplayer to the executable, for example,
`yarp connect /file/ch0dvs:o /yarp-example/AE:i`
