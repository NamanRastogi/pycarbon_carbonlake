## pycarbon: AI framework integration for Apache CarbonData

Supported AI framework:
- TensorFlow
- pytorch
- mxnet
- pyspark

By using pyarbon, data access from AI framework is accelerated up to 100X. 

## pycarbon install

$ git clone https://github.com/carbonlake/pycarbon.git

$ cd pycarbon

$ pip install . --user

## how to use

if you have a CarbonData dataset, you can use pycarbon to read data. For the generation of CarbonData dataset, you can see the examples:
`generate_external_dataset_carbon.py <https://github.com/HuaweiBigData/pycarbon/blob/master/examples/hello_world/external_dataset/generate_external_dataset_carbon.py>` and 
`generate_pycarbon_dataset.py <https://github.com/HuaweiBigData/pycarbon/blob/master/examples/hello_world/pycarbon_dataset/generate_pycarbon_dataset.py>`

#### PySpark and SQL
    # Create a dataframe object from carbon files
    spark.sql("create table readcarbon using carbon location '" + str(dataset_path) + "'")
    dataframe = spark.sql("select * from readcarbon")

    # Show a schema
    dataframe.printSchema()

    # Count all
    dataframe.count()

    # Show just some columns
    dataframe.select('id').show()

    # Also use a standard SQL to query a dataset
    spark.sql('SELECT count(id) from carbon.`{}` '.format(dataset_url)).collect()

some details are illustrated in `pyspark_hello_world_carbon.py <https://github.com/HuaweiBigData/pycarbon/blob/master/examples/hello_world/pycarbon_dataset/pyspark_hello_world_carbon.py>`

#### Tensorflow Dataset API
    with make_carbon_reader('file:///some/localpath/a_dataset') as reader:
        dataset = make_pycarbon_dataset(reader)
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            sample = sess.run(tensor)
            print(sample.id)

some details are illustrated in `tf_example_carbon.py <https://github.com/HuaweiBigData/pycarbon/blob/master/examples/mnist/tf_example_carbon.py>`

#### Pytorch API
    with DataLoader(make_carbon_reader('file:///localpath/mnist/train', num_epochs=10,
                            transform_spec=transform), batch_size=64) as train_    loader:
        train(model, device, train_loader, 10, optimizer, 1)

some details are illustrated in `pytorch_example_carbon.py <https://github.com/HuaweiBigData/pycarbon/blob/master/examples/mnist/pytorch_example_carbon.py>`
