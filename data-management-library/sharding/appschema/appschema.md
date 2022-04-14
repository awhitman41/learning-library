# Eshop Application Schema & Code Snippet (read-only)

## Introduction   
eShop application is server side rendered web application hosted on a NodeJS web server. Application is designed on MVC (Model, View and Controller) architecture pattern, with view as HTML 5 pages, controllers written in NodeJS to handle user http request and model to access database objects.

It utilizes different types of tables (relational, non-relational) in Oracle database to persist application data.

Relational table stores data related to **Customer** and their **Orders** while the **Product Catalog** and **Customer Reviews** data are stored as JSON in SODA collections.


*Estimated Lab Time*: 5 Minutes

### Objectives
In this lab, you will:
* Understand application code.
* Understand application connectivity.

### Prerequisites
This lab assumes you have:
- A Free Tier, Paid or LiveLabs Oracle Cloud account
- You have completed:
    - Lab: Prepare Setup (*Free-tier* and *Paid Tenants* only)
    - Lab: Environment Setup
    - Lab: Initialize Environment

## Task 1: Application Schema and Code Snippet

1. User makes a valid http request to server using a rest call from browser.

2. Application validates the route and map it to a proper controller as shown below:

    ```
    app.route("/shop/product/:key").get(shopController.fetchProductById);
    ```

3. Controller function pull out the provided details from the request and make pass it to model layer, as shown below in the code snippet:

    ```
    exports.fetchProductById = function(req,res,next)
        {
            let item = req.params.key;
            dbService.getProductByKey(item)
        }
    ```
dbService.getProductByKey(item) – calling model layer function with “item” as parameter.

4. Model layer which interacts with database, makes a no-SQL SODA API call or SQL call to access data from SODA Collections or Relational tables/SODA Collections respectively.

   **Connect to database shard using sharding key**

    ```
    connection = await oracledb.getConnection({

    user: 'SHARDUSERTEST',
    password: 'oracle',
    connectString: '158.101.120.251:1522/oltp_rw_products.orasdb.oradbcloud',
    shardingKey:[id]

    });
    ```

	**Query the SODA Collection or Table (In this case PRODUCTS is a SODA Collection)**

	```
    const soda = connection.getSodaDatabase();
    const collection = await soda.openCollection("PRODUCTS");
    const doc = await collection.find().key(id).getOne();
    const content = doc.getContent();

    ```

5. Views (HTML page) get created using the data returned to controller from model layer, as show below and then sent to client as response.

    ```
   dbService.getProductByKey(item)
    	.then(
        (data)=>{
            res.render('productInfo',{product:data});
            res.end();
        },
        err=>{
            return next(err);
        }
    	)
    ```
## Task 2: Application Connection Details

In Oracle Sharding, database query and DML requests are routed to the shards in two main ways, depending on whether a sharding key is supplied with the request. These two routing methods are called **proxy routing** and **direct routing.**

**Proxy Routing:** Queries that need data from multiple shards, and queries that do not specify a sharding key, cannot be routed directly by the application. Those queries require a proxy to route requests between the application and the shards. Proxy routing is handled by the shard catalog query coordinator.

Example: Database connection details:

			module.exports =
			{
               sharding: {
      			  user: 'SHARDUSERTEST',
       			  password: 'oracle',
      			  connectString: '10.0.20.102:1521/cat1pdb',
        				poolMin: 10,
        				poolMax: 10,
        				poolIncrement: 0
  			  }

**Direct Routing:** You can connect directly to the shards to execute queries and DML by providing a sharding key with the database request. Direct routing is the preferred way of accessing shards to achieve better performance, among other benefits.

Example: Database connection details by passing a sharding key:

			connection = await oracledb.getConnection({
            			user: 'SHARDUSERTEST',
             			password: 'oracle',
             			connectString: '10.0.20.101:1521/oltp_rw_products.shardcatalog1.oradbcloud',
              			shardingKey:[docmt.PRODUCT_ID]
            			  });

For more details for the eShop code snippet click [here] (https://github.com/nishakau/ShardingSampleCode.git)

You may now proceed to the next lab.

## Learn More

- [Oracle JSON Developers Guide 19c] (https://docs.oracle.com/en/database/oracle/oracle-database/19/adjsn/index.html)
- [Introduction to SODA] (https://docs.oracle.com/en/database/oracle/simple-oracle-document-access/adsdi/overview-soda.html#GUID-BE42F8D3-B86B-43B4-B2A3-5760A4DF79FB)

## Acknowledgements
* **Authors** - Shailesh Dwivedi, Database Sharding PM , Vice President
* **Contributors** - Balasubramanian Ramamoorthy , Alex Kovuru, Nishant Kaushik, Ashish Kumar, Priya Dhuriya, Richard Delval, Param Saini,Jyoti Verma, Virginia Beecher, Rodrigo Fuentes
* **Last Updated By/Date** - Priya Dhuriya, Staff Solution Engineer - July 2021
