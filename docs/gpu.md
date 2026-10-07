# Running on a GPU (Nebius GPU VM)

ReproFix can run the repository's commands in a sandbox container that can see the VM's NVIDIA GPU, and it records which
GPU that was, with the verbatim `nvidia-smi` output, in the report. This page says what that does and does not prove, and
how to set it up on a Nebius AI Cloud GPU VM.

## What has and has not been checked

| | State |
|---|---|
| `REPROFIX_SANDBOX_GPU=1` adds `--gpus all` to the run phase only (never to the install phase) | Unit-tested: the exact `docker run` command line |
| The probe runs `nvidia-smi` with the same container flags as a run (`--gpus all`, no network, read-only root, all capabilities dropped) and reads the answer | Unit-tested with a recording fake and with `nvidia-smi` text written from its documented output formats (`tests/test_hardware.py`) |
| A GPU appears in the report (`hardware`), in the UI (Report tab) and in `reprofix doctor` | Unit-tested and UI-tested with synthetic records |
| **A real GPU, a real NVIDIA driver, the NVIDIA Container Toolkit, or `--gpus all` actually working** | **Not tested. No GPU hardware was available while this was written.** |
| **What a real H100 / L40S host prints**, and whether the parser handles every driver's output | **Not tested.** The parser skips lines it does not understand and the report then says "no GPU visible"; it never invents one. |
| A Nebius GPU VM was created, or ReproFix was run on one | **No.** The commands below are copied from Nebius's and NVIDIA's documentation (linked), not run. |
| A paper's result was reproduced on a GPU | **No.** The repository's own benchmark tasks are CPU-only toys. |

## What the record proves, and what it does not

At the start of every run (before the baseline) ReproFix asks the sandbox which hardware the run phase can see:

* Docker backend with `REPROFIX_SANDBOX_GPU=1`: it starts a throwaway container exactly like a run container and runs
  `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits`, then plain `nvidia-smi`
  for the CUDA version in its header.
* Docker backend without the flag: no container is started and the report says "CPU only (GPU not enabled)".
* Development (`local`) backend: it runs the host's `nvidia-smi` if there is one. That backend has no device isolation.

The report's `hardware` object has the status (`detected`, `none_visible`, `probe_failed`, `not_requested`), the parsed GPUs,
the driver and the driver's maximum CUDA version, where the probe ran, and the raw output (first 1,500 characters).
If `REPROFIX_SANDBOX_GPU=1` is set and no GPU is found, the report carries a caveat saying so.

**It shows which GPU the sandbox can see. It does not show that the repository's code ran on it.** A script that never
moves a tensor to `cuda` runs on the CPU even on an H100 host. If you need that proof, have the repository print
`torch.cuda.get_device_name(0)` (or similar) next to its metric; the Output tab shows the run's stdout.

## Setting it up on a Nebius AI Cloud GPU VM

Everything here is from the documentation pages linked in each step. None of it was run.

**1. Create the VM.** Nebius's [Compute quickstart](https://docs.nebius.com/compute/quickstart) creates a GPU VM with the
CLI. The platform and preset names below are the ones its pages use; run `nebius compute platform list` to see what your
project can create (GPU quotas may need a request first).

```bash
export SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
export USER_DATA=$(jq -Rrs '.' <<EOF
#cloud-config
users:
  - name: user
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub)
EOF
)
export VM_ID=$(nebius compute instance create \
  --name reprofix-gpu \
  --resources-platform gpu-h100-sxm \
  --resources-preset 1gpu-16vcpu-200gb \
  --boot-disk-managed-disk-name reprofix-gpu-disk \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 300 \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
  --boot-disk-attach-mode READ_WRITE \
  --cloud-init-user-data "$USER_DATA" \
  --network-interfaces "[{\"name\": \"eth0\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": {}}]" \
  --format jsonpath='{.metadata.id}')
export VM_IP=$(nebius compute instance get --id $VM_ID --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]')
ssh user@$VM_IP
```

The `ubuntu24.04-cuda13.0` image family is the one Nebius's quickstart uses, and the page says NVIDIA drivers come
pre-installed with it. The 300 GiB disk is my choice, not Nebius's: a PyTorch repository's dependencies take about 5 GB
per run (see [deployment.md](deployment.md)). Preemptible ("spot") VMs exist for GPU platforms
([docs](https://docs.nebius.com/compute/virtual-machines/preemptible)): Compute may stop them at any time, so they suit
benchmark runs but not a server people are using.

**2. Check the driver.** On the VM:

```bash
nvidia-smi
```

You should see the GPU table. The header's `CUDA Version` is the newest CUDA the driver supports, not something installed
(the image family's name suggests 13.0, but I did not see its output).

**3. Install Docker, then the NVIDIA Container Toolkit.** Docker Engine: [install instructions](https://docs.docker.com/engine/install/).
Then, from NVIDIA's
[install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (Ubuntu/Debian
with apt; the guide pins a toolkit version, so use the one it shows, and read the guide in case these steps have changed):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**4. Build ReproFix's sandbox image and test a GPU container.** From the ReproFix checkout (see
[deployment.md](deployment.md) for installing ReproFix itself):

```bash
docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest .
docker run --rm --gpus all --network none reprofix-sandbox:latest nvidia-smi
```

The second command is what the probe does. If it prints the GPU table, containers can see the GPU. If it fails with
`could not select device driver "" with capabilities: [[gpu]]`, the toolkit is not installed or Docker was not restarted
after `nvidia-ctk runtime configure`. If it says `nvidia-smi: executable file not found`, the toolkit is not injecting the
driver utilities into containers.

**5. Turn it on and check.**

```bash
export REPROFIX_SANDBOX_GPU=1        # or put it in .env / /etc/reprofix.env
reprofix doctor                      # the Sandbox section has a "GPU in the sandbox" line
```

Then run as usual. The Report tab shows a **Hardware record** with the verbatim `nvidia-smi` output, and a paper-claims card
says which hardware the sandbox could see.

**6. Stop paying when you are done.** A GPU VM is billed while it exists; check Nebius's
[pricing](https://nebius.com/prices) and your balance before leaving one running. Deleting it:

```bash
nebius compute instance delete --id $VM_ID
nebius compute disk list          # Nebius's CLI reference says instance delete also deletes managed disks declared in the
                                  # instance spec; confirm nothing is left, and delete leftovers with: nebius compute disk delete --id <id>
```

## Things to know before trusting a GPU number

* **GPUs make results less repeatable.** Many CUDA kernels are non-deterministic, TF32/fp16 settings change numerics, and a
  different GPU model than the paper's can move a metric. The claims card's default tolerance (±1 percentage point) is a
  chosen default, not a measured property of your hardware: run the same repository twice and look at the spread before
  deciding a miss is a real miss.
* **`--gpus all` gives every run every GPU.** Two overlapping runs share the device and its memory. Set
  `REPROFIX_MAX_CONCURRENT_RUNS=1` when you need clean timings or memory.
* **GPU passthrough widens the sandbox.** The container gets the NVIDIA device nodes and driver libraries, a larger surface
  than the CPU-only sandbox. Untrusted repositories are still the main risk; see [security.md](security.md). The run phase
  still has no network, and `--gpus all` is never applied to the install phase.
* **PyTorch.** The `torch` wheels on PyPI for Linux x86-64 are CUDA builds that pull NVIDIA's CUDA runtime packages (also
  from PyPI) as dependencies, so a host driver plus the container toolkit should be enough. That is an expectation from how
  those wheels are packaged; it has not been tried here. Installing them through the egress proxy needs only `pypi.org` and
  `files.pythonhosted.org`. The `download.pytorch.org` index (used for CPU-only wheels) is not on the default allow-list.
* **Cost.** ReproFix's model calls (Nebius Token Factory) and the GPU VM are billed separately. A 30-minute run of a small
  repository costs the model tokens plus 30 minutes of VM time, on a VM you would otherwise leave idle.
