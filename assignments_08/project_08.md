# Project 08 - Azure Intro and Cost Analysis

**Video link:** https://youtu.be/GNjL3FjNEh0

---

## Cost Analysis Write-Up

### Scenario A - Lightweight Compute

Scenario A uses a Standard_B1s VM (1 vCPU, 1 GB RAM) running 160 hours a month (8 hours a day, 5 days a week). At roughly $0.0104/hour, the monthly VM cost comes out to about **$1.66**. That number is lower than I expected. For less than two dollars a month in compute time, you can run a scheduled lightweight pipeline on actual cloud infrastructure. For small internal tools or scheduled jobs that do not need to run around the clock, the cost is almost negligible.

### Scenario B - Heavy Analytics Workload

Scenario B includes three components running simultaneously for a full month (730 hours):

- **Standard_NC6s_v3 VM** (6 vCPU, 1 V100 GPU): approximately $3.06/hour, coming out to **$2,233.80/month** for the VM alone.
- **Azure SQL Database** (General Purpose, 4 vCores): **$741.16/month**.
- **Azure Blob Storage** (1 TB of data): **$21.84/month**.

Total for Scenario B: **$2,996.80/month**.

The GPU VM cost is what stands out. The NC6s_v3 runs roughly 135x more expensive per hour than the B1s, and that gap compounds fast when you are running it 24/7 versus just business hours. A GPU VM left always-on for a month costs more than $2,000 before you add anything else. The Azure SQL Database cost also surprised me at $741/month for General Purpose with 4 vCores, which is nearly a third of the total bill on its own. That puts both components firmly in the category of "right-size carefully and shut down when not in use." For workloads that only need the GPU during training runs, reserved instances or spot pricing would significantly reduce that cost.

### Additional Exploration

While exploring the calculator beyond the two required scenarios, I added an Azure Kubernetes Service node pool and an Azure Machine Learning workspace to see what a more complete ML pipeline might look like. The AKS cluster added another few hundred dollars depending on node size, and the Azure Machine Learning workspace itself is free but the underlying compute it manages is billed separately. It makes sense once you understand the layering, but the pricing calculator does a good job of making those dependencies visible before you commit to anything.

I also looked at foundary tools(Azure AI) out of curiosity. Some of the API-based services like AI Custom Vision and Azure Language are priced per transaction rather than per hour, which is a fundamentally different cost model than compute. For high-volume use, that can add up faster than a VM would.

### Script Output

```
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.66
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 1345.7x more than Scenario A
```

The calculated costs matched what the Pricing Calculator showed for the VM components. The multiplier on that last line is accurate but still surprising to see as a raw number. It reinforces why right-sizing your infrastructure matters. Choosing the wrong VM tier for a workload and leaving it running is one of the fastest ways to blow a cloud budget.