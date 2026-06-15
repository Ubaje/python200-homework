# Week 8 Warmup - Cloud Computing

## Cloud Concepts

### Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go: instead of purchasing physical hardware upfront, you rent compute, storage, and networking from a provider and pay only for what you use. Owning your own servers requires large capital expenditures upfront, ongoing maintenance costs, and forces you to provision for peak load even when that capacity sits idle most of the time. The cloud shifts that to an operational expense that scales with actual demand.

### Cloud Concepts Question 2

Vertical scaling means upgrading a single machine to be more powerful, giving it a better CPU, more RAM, or a better GPU. Horizontal scaling means adding more machines and distributing the workload across them.

You would choose vertical scaling when the work cannot easily be parallelized and just needs a faster or larger machine. You would choose horizontal scaling when the workload can be split into independent chunks and one machine is simply not enough to keep up with demand.

**Scenarios:**

- *Web app going from 1,000 to 100,000 users after a viral launch:* Horizontal scaling. Web requests are stateless and independent, so spinning up additional instances behind a load balancer is the natural fit here.
- *Data scientist's model training job running too slowly, wants more GPU and RAM:* Vertical scaling. Training a single model is not easily parallelized across separate machines, so upgrading to a more powerful VM is the right move.
- *Data pipeline scaling from 10 to 10,000 files, work can be split across machines:* Horizontal scaling. The assignment explicitly says the work can be distributed, so adding more worker nodes to process files in parallel is the correct approach.

### Cloud Concepts Question 3

**Classifications:**

- **Gmail** - SaaS. It is a fully managed application you access through a browser; you manage nothing except your own emails.
- **Azure Virtual Machines** - IaaS. You get a raw virtual machine and are responsible for the OS, software installs, and configuration.
- **Azure App Service** - PaaS. You deploy your code and the platform handles the underlying infrastructure and scaling.
- **AWS S3** - IaaS. It is a managed storage primitive, but you configure bucket policies, access controls, and data organization yourself.
- **GitHub Codespaces** - PaaS. The platform provisions a containerized dev environment for you; you bring your code and config.
- **Snowflake** - SaaS (or managed data platform). It is a fully managed data warehouse delivered as a service; you write queries and manage your data but touch no infrastructure.

**Definitions:**

*IaaS (Infrastructure as a Service)* gives you the raw building blocks: virtual machines, storage, and networking. You are responsible for everything above the hardware, including the operating system, runtime, security patches, and application software. Example: Azure Virtual Machines. As a developer, you provision the VM, choose the OS, install your dependencies, and manage the machine going forward.

*PaaS (Platform as a Service)* handles the infrastructure for you, leaving you responsible only for your application code and its configuration. The platform manages the OS, runtime, and scaling. Example: Azure App Service. You deploy your app and let Azure worry about the servers underneath it.

*SaaS (Software as a Service)* is a fully managed application you access and use. You are responsible for essentially nothing technical. Example: Gmail. You log in, write emails, and Google handles everything else.

### Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake is a specialized layer built on top of a cloud provider that pre-configures the infrastructure specifically for data and analytics workloads. Instead of assembling your own stack from raw cloud primitives (compute VMs, storage buckets, orchestration tools), the platform handles all of that for you, optimized out of the box for things like large-scale data processing or SQL analytics.

What you gain is speed and simplicity. You can start running distributed Spark jobs or querying petabytes of data without having to configure clusters, networking, or storage integrations yourself. What you give up is flexibility and some cost efficiency. You are locked into the platform's abstractions, and since the platform itself runs on top of the cloud provider, you are paying for that extra layer of managed service on top of the underlying infrastructure costs.

### Cloud Concepts Question 5

The two situations where the cloud is probably not the right choice are:

1. Your dataset fits comfortably on a single machine and your compute demands are modest. Local processing will often be faster and cheaper than standing up cloud infrastructure for a workload that does not need it.
2. The learning curve and complexity of the ecosystem would create more friction than value for the task at hand. Cloud platforms are enormous and setting up even simple things can take considerable time when you are new to them.

---

## Azure Basics

### Azure Basics Question 1

An Azure subscription is the top-level billing account that owns all the resources within an organization. CTD has one subscription that the whole cohort shares. A resource group is a logical container within that subscription that bundles related resources together for a specific project or user. Each student gets their own personal resource group, so the resource group is yours alone, while the subscription belongs to CTD.

### Azure Basics Question 2

Cloud Shell being ephemeral by default means that every time you close the shell session, all files you created are deleted and the container resets to a clean state. Nothing persists between sessions unless you wire up external storage. The course setup addresses this by mounting a persistent Azure file share to your Cloud Shell home directory, so files created under `~` (SSH keys, scripts, config) survive across sessions.

### Azure Basics Question 3

An SSH private key is the secret half of the key pair that lives only on your machine and should never be shared. The public key is the non-secret half that you upload to any remote system you want to connect to. When you initiate a connection, SSH uses cryptographic math to verify that your private key corresponds to the public key on the server, proving your identity without your password ever crossing the network. It is safe to share the public key because knowing it gives an attacker no information about the private key, and the private key cannot be derived from it.

### Azure Basics Question 4

**Without `--output table`:**

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#kezzy02@gmail.com",
    "type": "user"
  }
```

Adding `--output table` reformats the JSON response into a human-readable table with column headers: Name, Location, Status; stripping away the nested structure and only showing a quick peek on the azure account details.