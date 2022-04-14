# Introduction

## About this Workshop
This workshop introduces you to Oracle Database backup and recovery with the Recovery Manager (RMAN). After completing this workshop, you should be familiar with the basic concepts of Oracle Database backup and recovery operations, know how to implement a disk-based backup strategy, and perform repairs to database files.

Estimated Workshop Time: 2 hours

### Objectives
In this workshop, you will learn the following Oracle Database activities.
- Configure recovery settings
- Configure backup settings
- Perform and schedule backups
- Manage backups
- Use data recovery advisor to repair failures
- Rewind a table using Flashback Table
- Recover a dropped table using Flashback Drop

### Prerequisites
A Free Tier, Paid or LiveLabs Oracle Cloud account  

>Note: If you have a **Free Trial** account, when your Free Trial expires your account will be converted to an **Always Free** account. You will not be able to conduct Free Tier workshops unless the Always Free environment is available. [Click here for the Free Tier FAQ page](https://www.oracle.com/cloud/free/faq.html).


## Appendix 1: Overview of Oracle Database Backup and Recovery
Oracle Database backup and recovery operations focus is on the physical backup of database files, which permits you to reconstruct your Oracle Database in case of a failure/corruption.

Oracle Recovery Manager (RMAN), a command-line tool, is the method preferred by Oracle for efficiently backing up and recovering your Oracle Database. The files protected by the backup and recovery facilities built into RMAN include data files, control files, server parameter files, and archived redo log files. With these files, you can reconstruct your Oracle Database. RMAN is designed to work intimately with the server, providing block-level corruption detection during backup and restore. RMAN optimizes performance and space consumption during backup with file multiplexing and backup set compression and integrates with leading tape and storage media products. The backup mechanisms work at the physical level to protect against file damage, such as the accidental deletion of a data file or the failure of a disk drive. RMAN can also be used to perform point-in-time recovery to recover from logical failures when other techniques such as flashback cannot be used.

The Oracle Flashback features provide a range of physical and logical data recovery tools as efficient, easy-to-use alternatives to physical and logical backups. The Oracle Flashback features enable you to reverse the effects of unwanted database changes without restoring data files from backup.

Click on the next lab to **Get Started**.


## Learn More

[Blog on Introducing Oracle Database 21c](https://blogs.oracle.com/database/introducing-oracle-database-21c)

[Oracle Database Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/index.html)

[Oracle Cloud Infrastructure Documentation](https://docs.oracle.com/en-us/iaas/Content/Identity/Concepts/overview.htm)


## Acknowledgements
- **Author**: Suresh Mohan, Database User Assistance Development Team
- **Contributors**: Suresh Rajan, Manish Garodia, Subhash Chandra, Ramya P
- **Last Updated By & Date**: Suresh Mohan, February 2022
