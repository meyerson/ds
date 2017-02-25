from sql_help import *

PANDAS_TO_HIVE_DTYPES = {'flo':'FLOAT', 'int':'INT', 'uin':'INT', 'boo':'BOOLEAN', 'dat':'DATE', 'obj':'STRING'}
def make_create_hive_statement_for_dataframe(df):
    # maps first 3 chars of python/pandas datatype to Hive datatype
    return ','.join('%s %s' % (name,PANDAS_TO_HIVE_DTYPES[str(type)[:3]]) for name, type in zip(df.columns, df.dtypes))

PANDAS_TO_SQL_DTYPES = {'flo':'FLOAT', 'int':'INT', 'boo':'BOOLEAN', 'dat':'DATE', 'obj':'TEXT'}
def make_create_sql_statement_for_dataframe(df):
    return ','.join('%s %s' % (name,PANDAS_TO_SQL_DTYPES[str(type)[:3]]) for name, type in zip(df.columns, df.dtypes))

class TableWriter:

    from sql import write_frame
    def __init__(self, conn_str, table_name, flav='sqlite', if_exists='append'):
        self.conn_str = conn_str
        self.table_name = table_name
        self.flav = flav
        self.if_exists = if_exists
        
    def __call__(self, data, index_cols=None):
        if self.flav == 'hive':
            import hs2
            hs2.Hive(self.conn_str).write_frame(data, self.table_name, if_exists=self.if_exists)
        else:
            cnx = db_connect(self.conn_str, self.flav)

            if self.if_exists == 'replace':
                # Not appending; drop the current table if it exists.
                query = "DROP TABLE IF EXISTS %s" % (self.table_name.lower(),)
                execute(query, cnx)
            
            # Create table if it doesn't exist.
            column_definitions = make_create_sql_statement_for_dataframe(data)
            query = "CREATE TABLE IF NOT EXISTS %s (%s)" % (self.table_name.lower(), column_definitions.lower())
            execute(query, cnx)
            
            from sql import write_frame
            write_frame(data, self.table_name, cnx, self.flav, self.if_exists)
            if index_cols != None:
                non_qualified_name = self.table_name.split('.')[-1]
                if self.flav=='sqlite':
                    execute('CREATE INDEX IF NOT EXISTS %s_index on %s (%s)' % (non_qualified_name, self.table_name, ','.join(index_cols)), cnx);
                else:
                    execute('CREATE INDEX %s_index on %s (%s)' % (non_qualified_name, self.table_name, ','.join(index_cols)), cnx);
