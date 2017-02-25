import pyhs2

with pyhs2.connect(host='azuscdbdm01',
                   port=10000,
                   authMechanism="PLAIN",
                   user='bigdata',
                   password='B1Gd@t@',
                   database='productiondata') as conn:
    '''
    Simple example of connecting to and returning results from a hive DB.
    '''
    with conn.cursor() as cur:
    	#Show databases
    	print cur.getDatabases()

    	#Execute query
        cur.execute("select * from LISTRUNTICKETTYPETB")

        #Return column info from query
        print cur.getSchema()
	data = cur.fetchall ()
	print data
