"""Tests for MqttClient connection state handling.

We don't stand up a real broker. We build the client, then invoke its
on_connect / on_disconnect callbacks directly with the reason codes paho
would deliver, and verify the internal `connected` flag and discovery side
effects behave correctly.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from watermeter import Config, MqttClient


@pytest.fixture
def cfg():
    return Config(
        esp32_base_url="http://localhost",
        mqtt_host="127.0.0.1",
        mqtt_port=1883,
        mqtt_main_topic="home/watermeter",
        ha_discovery_prefix="homeassistant",
    )


@pytest.fixture
def logger():
    return logging.getLogger("test-mqtt")


@pytest.fixture
def client(cfg, logger):
    # Patch paho's Client to avoid touching the network.
    with patch("watermeter.mqtt.Client") as ClientCls:
        ClientCls.return_value = MagicMock()
        c = MqttClient(cfg, logger)
        yield c


class TestOnConnectReasonCode:
    def test_success_sets_connected(self, client):
        # paho passes the on_connect callback as client.on_connect after init.
        on_connect = client.client.on_connect
        assert on_connect is not None

        assert client.connected is False
        on_connect(client.client, None, {}, 0)
        assert client.connected is True

    def test_success_publishes_discovery(self, client):
        on_connect = client.client.on_connect
        on_connect(client.client, None, {}, 0)
        # discovery() calls self.client.publish() multiple times (total, rate, rate_lpm, camera).
        assert client.client.publish.call_count >= 4

    def test_failure_keeps_disconnected(self, client):
        on_connect = client.client.on_connect
        # Reason code 5 == "not authorized" in MQTT v3.
        on_connect(client.client, None, {}, 5)
        assert client.connected is False

    def test_failure_does_not_publish_discovery(self, client):
        on_connect = client.client.on_connect
        client.client.publish.reset_mock()
        on_connect(client.client, None, {}, 5)
        assert client.client.publish.call_count == 0


class TestOnDisconnect:
    def test_disconnect_clears_connected(self, client):
        # First connect successfully, then simulate a disconnect.
        client.client.on_connect(client.client, None, {}, 0)
        assert client.connected is True

        client.client.on_disconnect(client.client, None, {}, 0)
        assert client.connected is False


class TestPublishGatedByConnection:
    def test_publish_noops_when_disconnected(self, client):
        assert client.connected is False
        client.client.publish.reset_mock()
        client.publish("some/topic", "hello")
        assert client.client.publish.call_count == 0

    def test_publish_forwards_when_connected(self, client):
        client.client.on_connect(client.client, None, {}, 0)
        client.client.publish.reset_mock()
        client.publish("some/topic", "hello")
        client.client.publish.assert_called_once_with("some/topic", "hello", retain=False)
