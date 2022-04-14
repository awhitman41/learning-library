# Testing the Deployment of Application to the Oracle Container Engine for Kubernetes (OKE) on Oracle Cloud Infrastructure (OCI) 

## Introduction

In this lab, we access the application *opdemo* and verify the successful migration of an offline on-premise domain. we also verify the loadbalancing between managed server pods. Later, we use WebLogic Remote Console, to verify the successful deployment of resources of test-domain in kubernetes environment.


### About WebLogic Remote Console

The WebLogic Remote Console is a lightweight, open source console that you can use to manage your WebLogic Server domain running anywhere, such as on a physical or virtual machine, in a container, Kubernetes, or in the Oracle Cloud. The WebLogic Remote Console does not need to be colocated with the WebLogic Server domain.

You can install and run the WebLogic Remote Console anywhere, and connect to your domain using WebLogic REST APIs. You simply launch the desktop application and connect to the Administration Server of your domain. Or, you can start the console server, launch the console in a browser and then connect to the Administration Server.

The WebLogic Remote Console is fully supported with WebLogic Server 12.2.1.3, 12.2.1.4, and 14.1.1.0.

**Key Features of the WebLogic Remote Console**

* Configure WebLogic Server instances and clusters
* Create or modify WDT metadata models
* Configure WebLogic Server services, such as database connectivity (JDBC), and messaging (JMS)
* Deploy and undeploy applications
* Start and stop servers and applications
* Monitor server and application performance

### Objectives

In this lab, you will:

* Access the Application through the Browser and verify the load balancing between two managed server pods.
* Connect to Admin Server using WebLogic Remote Console and explore the WewbLogic Domain and its resources.

### Prerequisites

* Successfully created the virtual machine, which consist of all required softwares.

## Task 1: Access the Application through the Browser

In this task, we access the *opdemo* application. We click on refresh icon to make multiple request to Application, to verify loadbalancing between two managed servers.

1. Copy the below URL and replace *XX.XX.XX.XX* with your IP, which is populated in last lab. You can see the below output.
    ```bash
    <copy>http://XX.XX.XX.XX/opdemo/?dsname=testDatasource</copy>
    ```
    ![Open Application](images/OpenApplication.png)


2. If you click on Refresh icon, You can see load balancing between two managed server pods.
    ![Show Loadbalancing](images/ShowLoadbalancing.png)


## Task 2: Connection to Admin Server using WebLogic Remote Console

In this task, we explore the WebLogic Remote Console. We create connection to *Admin Server* in Remote Console and verify the resources in WebLogic Domain. This verifies the successful migration of an on-premise domain into the Oracle Kubernetes Cluster. 

1. Go back to WebLogic Remote Console, Click on *Activities*, then select the *WebLogic Remote Console* Icon.
    ![Open Remote Console](images/OpenRemoteConsole.png)

2. Select *Add Admin Server Connection Provider* and click *Choose*.
    ![Admin Server Connection](images/AdminServerConnection.png)

3. Enter the following data and click *OK*.<br>
    Connection Provider Name: Admin<br> Server<br>
    Username: weblogic<br>
    Password: welcome1<br>
    URL:  `Copy_Public_IP_From_WKTUI`</br>
    ![Connection Details](images/ConnectionDetails.png)

4. Click on *Edit Tree* icon, then Select *Services* -> *JDBC System Resources*. You can observe the same Datasouce, which we had seen in on-premise domain.
    ![Verify Datasources](images/VerifyDatasources.png)

5. Click on *Monitoring Tree* Icon as shown then select *Running Servers*. You can see we have *Admin Server* and 2 Managed Server pods are running. If you use *Scroll* Button in down side, you can see WebLogic Versin is *12.2.1.3.0*.
    ![Running Servers](images/RunningServers.png)
    ![WebLogic Version](images/WebLogicVersion.png)


## Acknowledgements

* **Author** -  Ankit Pandey
* **Contributors** - Maciej Gruszka, Sid Joshi
* **Last Updated By/Date** - Kamryn Vinson, March 2022