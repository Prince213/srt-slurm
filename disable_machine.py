#!/usr/bin/env python3
"""
Script to find machines by hostname from HOSTNAMES list and disable them using GTL API.
"""
HOSTNAMES = [
    "PRENYX-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "CW-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "EOS-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "PTYCHE-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "LYRIS-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "GANON-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "MY_CLUSTER_NAME-CUSTOM-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "COLORADO-DEBUG-SC-FAT-EHPC-ICL8380-801",
    "GANON-DEBUG-LUNA80-CR-41",
    "CW-DEBUG-LUNA80-CR-41",
    "COLORADO-DEBUG-LUNA80-CR-41",
    "EOS-DEBUG-LUNA80-CR-41",
    "PTYCHE-DEBUG-LUNA80-CR-41",
    "PRENYX-DEBUG-LUNA80-CR-41",
    "LYRIS-DEBUG-LUNA80-CR-41",
    "LYRIS-DEBUG-ROGLIU-LOCAL-DEBUG",
    "CW-DEBUG-ROGLIU-LOCAL-DEBUG",
    "COLORADO-DEBUG-ROGLIU-LOCAL-DEBUG",
    "PTYCHE-DEBUG-ROGLIU-LOCAL-DEBUG",
    "EOS-DEBUG-ROGLIU-LOCAL-DEBUG",
    "GANON-DEBUG-ROGLIU-LOCAL-DEBUG",
    "PRENYX-DEBUG-ROGLIU-LOCAL-DEBUG",
    "COMPUTELAB-CUSTOM-DEBUG-ROGER-UBUNTU",
    "LYRIS-DEBUG-LOCAL",
    "GANON-DEBUG-LOCAL",
    "PTYCHE-DEBUG-LOCAL",
    "CUSTOM-DEBUG-LOCAL",
    "EOS-DEBUG-LOCAL",
    "PRENYX-DEBUG-LOCAL",
    "HSG-DEBUG-LUNA80-CR-40",
    "TOYAMA-CUSTOM-DEBUG-LUNA80-CR-40",
    "GB300-CUSTOM-DEBUG-LUNA80-CR-40",
    "GANON-DEBUG-ROGER-UBUNTU",
    "MY_CLUSTER_NAME-CUSTOM-DEBUG-ROGER-UBUNTU",
    "LYRIS-DEBUG-SC-FAT-EHPC-CSL52",
    "COLORADO-DEBUG-SC-FAT-EHPC-CSL52",
    "EOS-DEBUG-SC-FAT-EHPC-CSL52",
    "TOYAMA-CUSTOM-DEBUG-SC-FAT-EHPC-CSL52",
    "PTYCHE-DEBUG-SC-FAT-EHPC-CSL52",
    "CW-DEBUG-SC-FAT-EHPC-CSL52",
    "PRENYX-DEBUG-SC-FAT-EHPC-CSL52",
    "GANON-DEBUG-SC-FAT-EHPC-CSL52",
    "LYRIS-DEBUG-LUNA80-CR-40-1",
    "PTYCHE-DEBUG-LUNA80-CR-40",
    "EOS-DEBUG-LUNA80-CR-40",
    "LYRIS-DEBUG-LUNA80-CR-40",
    "CW-DEBUG-LUNA80-CR-40",
    "PRENYX-DEBUG-LUNA80-CR-40",
    "COREWAEVE-DFW-CUSTOM-DEBUG-ROGER-UBUNTU",
    "CUSTOM-DEBUG-ROGER-UBUNTU",
    "PRENYX-DEBUG-LUNA-DVT-46",
    "CW-DEBUG-LUNA-DVT-46",
    "EOS-DEBUG-LUNA-DVT-46",
    "LYRIS-DEBUG-LUNA-DVT-46",
    "PTYCHE-DEBUG-LUNA-DVT-46",
    "HSG-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "COLORADO-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "PTYCHE-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "LYRIS-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "EOS-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "CW-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "PRENYX-DEBUG-SC-FAT-EHPC-EMR8570-800",
    "TOYAMA-DEBUG-LUNA80-CR-40-1",
    "CUSTOM-PDX-DEBUG-SYS-SSG-UVIKG03-1",
    "CUSTOM-PDX-DEBUG-SYS-SSG-UVIKG03-2",
    "CUSTOM-PDX-DEBUG-LUNA80-CR-40-2",
    "CUSTOM-PDX-DEBUG-LUNA80-CR-40-1",
    "CUSTOM-DEBUG-ROGER-UBUNTU-1",
    "CUSTOM-DEBUG-ROGER-UBUNTU-2",
    "CW-DEBUG-LOCAL",
    "LYRIS-DEBUG-ROGER-UBUNTU",
    "GCN-LOCAL-DEBUG-ROGER-UBUNTU",
    "COLORADO-DEBUG-ROGER-UBUNTU",
    "CW-DEBUG-ROGER-UBUNTU",
    "CW-DEBUG-1",
    "PRENYX-DEBUG-FANEY-DT",
    "PTYCHE-DEBUG-FANEY-DT",
    "EOS-DEBUG-FANEY-DT",
    "PTYCHE-DEBUG-ROGER-UBUNTU",
    "PRENYX-DEBUG-ROGER-UBUNTU",
    "EOS-DEBUG-ROGER-UBUNTU",
    "SSG-DEBUG-1",
    "COMPUTE-LAB-DEBUG-1",
    "PTYCHE-DEBUG-2",
    "PTYCHE-DEBUG-1",
    "PRENYX-DEBUG-2",
    "PRENYX-DEBUG-1",
    "EOS-DEBUG-2",
    "EOS-DEBUG-1",
]


from gtl_api import GTLAPI


def find_and_disable_machines():
    """Find machines by hostname from HOSTNAMES list and disable them."""
    api = GTLAPI()
    machine_api = api.Machine

    print(f"Looking up machine IDs for {len(HOSTNAMES)} hostnames...")

    # Get machine IDs for each hostname individually
    # This handles cases where some hostnames might not exist
    target_machines = []
    missing_hostnames = []

    for hostname in HOSTNAMES:
        try:
            machine_id = machine_api.GetMachineIdFromHostname(hostname)
            if machine_id > 0:
                target_machines.append((machine_id, hostname))
                print(f"  Found: {machine_id} - {hostname}")
            else:
                missing_hostnames.append(hostname)
        except Exception as e:
            # Hostname not found or error occurred
            missing_hostnames.append(hostname)
            print(f"  ✗ Not found: {hostname}")

    if missing_hostnames:
        print(
            f"\n⚠ Warning: {len(missing_hostnames)} hostnames from HOSTNAMES list were not found:"
        )
        for hostname in sorted(missing_hostnames):
            print(f"  - {hostname}")

    if not target_machines:
        print("\n✓ No machines matching hostnames in HOSTNAMES list found.")
        return

    print(
        f"\nFound {len(target_machines)} machines matching hostnames in HOSTNAMES list:"
    )
    for machine_id, hostname in target_machines:
        print(f"  - {machine_id}: {hostname}")

    print("\nDisabling machines...")

    # Disable each machine
    disabled_count = 0
    failed_count = 0
    for machine_id, hostname in target_machines:
        try:
            result = machine_api.DisableMachine(machine_id)
            print(f"✓ Disabled machine {machine_id}: {hostname}")
            disabled_count += 1
        except Exception as e:
            print(f"✗ Error disabling machine {machine_id} ({hostname}): {e}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(
        f"Completed: Disabled {disabled_count} out of {len(target_machines)} machines"
    )
    if failed_count > 0:
        print(f"Failed to disable: {failed_count} machines")
    if missing_hostnames:
        print(f"Hostnames not found: {len(missing_hostnames)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    find_and_disable_machines()
