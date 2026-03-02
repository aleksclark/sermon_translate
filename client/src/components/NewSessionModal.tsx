import { useEffect, useState } from "react";
import { Button, Group, Modal, NativeSelect, Stack, Text, TextInput } from "@mantine/core";
import { createSession, fetchPipelines } from "../api/index.ts";
import type { PipelineInfo, Session } from "../api/index.ts";
import { useAudioDevices } from "../hooks/useAudioDevices.ts";

export function NewSessionModal({
  opened,
  onClose,
  onCreated,
}: {
  opened: boolean;
  onClose: () => void;
  onCreated: (session: Session, inputDeviceId: string, outputDeviceId: string) => void;
}) {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState("");
  const [label, setLabel] = useState("");
  const [inputDevice, setInputDevice] = useState("");
  const [outputDevice, setOutputDevice] = useState("");
  const [creating, setCreating] = useState(false);
  const { inputs, outputs } = useAudioDevices();

  useEffect(() => {
    if (!opened) return;
    fetchPipelines().then((p) => {
      setPipelines(p);
      if (p.length > 0) setSelectedPipeline(p[0].id);
    });
  }, [opened]);

  useEffect(() => {
    if (inputs.length > 0 && !inputDevice) setInputDevice(inputs[0].deviceId);
  }, [inputs]);

  useEffect(() => {
    if (outputs.length > 0 && !outputDevice) setOutputDevice(outputs[0].deviceId);
  }, [outputs]);

  const handleCreate = async () => {
    if (!selectedPipeline) return;
    setCreating(true);
    try {
      const session = await createSession({
        pipeline_id: selectedPipeline,
        label: label || undefined,
      });
      setLabel("");
      onCreated(session, inputDevice, outputDevice);
      onClose();
    } finally {
      setCreating(false);
    }
  };

  const selected = pipelines.find((p) => p.id === selectedPipeline);

  return (
    <Modal opened={opened} onClose={onClose} title="New Session" centered>
      <Stack gap="sm">
        <NativeSelect
          label="Pipeline"
          data={pipelines.map((p) => ({ value: p.id, label: p.name }))}
          value={selectedPipeline}
          onChange={(e) => setSelectedPipeline(e.currentTarget.value)}
        />
        {selected && (
          <Text size="xs" c="dimmed">
            {selected.description}
          </Text>
        )}
        <TextInput
          label="Label (optional)"
          placeholder="My session"
          value={label}
          onChange={(e) => setLabel(e.currentTarget.value)}
        />
        <NativeSelect
          label="Audio Input"
          data={
            inputs.length > 0
              ? inputs.map((d) => ({ value: d.deviceId, label: d.label }))
              : [{ value: "", label: "No devices found" }]
          }
          value={inputDevice}
          onChange={(e) => setInputDevice(e.currentTarget.value)}
        />
        <NativeSelect
          label="Audio Output"
          data={
            outputs.length > 0
              ? outputs.map((d) => ({ value: d.deviceId, label: d.label }))
              : [{ value: "", label: "No devices found" }]
          }
          value={outputDevice}
          onChange={(e) => setOutputDevice(e.currentTarget.value)}
        />
        <Group justify="flex-end" mt="xs">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleCreate} loading={creating} disabled={!selectedPipeline}>
            Start Session
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
