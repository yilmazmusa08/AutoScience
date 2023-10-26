import React from "react";
import { InboxOutlined, LoadingOutlined } from "@ant-design/icons";
import {
  Row,
  Col,
  Button,
  Typography,
  Form,
  message,
  Upload,
  Select,
  InputNumber,
  Checkbox,
  Input,
} from "antd";
import { useApp } from "../../context/app.context";
import { isFileSizeWithinLimit } from "../../utils/validations";
import constant from "../../constants";
import {
  OperationType,
  operationTypeData,
  methodTypeData,
  transformTypeData,
  scalerTypeData,
} from "./types/featureEngineering";
import Papa from "papaparse";
import "./index.css";

const { Dragger } = Upload;

const FeatureEngineering: React.FC = () => {
  const {
    runFeatureEngineering,
    updateFile,
    file,
    updateTargetColumns,
    targetColumns,
    loadingFeatureEngineering,
  } = useApp();
  const [form] = Form.useForm();
  const operation = Form.useWatch("operation", form);

  const onFinish = (formData: any) => {
    if (file) {
      runFeatureEngineering({ file, ...formData });
    }
  };

  const PerformOperation = () => (
    <Row gutter={8}>
      <Col span={4}>
        <Form.Item
          label="Method"
          name="method"
          rules={[
            {
              required: true,
              message: "Please select a method!",
            },
          ]}
        >
          <Select
            placeholder="Select Method"
            disabled={!file}
            options={methodTypeData}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={10}>
        <Form.Item label="Col 1" name="col1">
          <Select
            allowClear
            placeholder="Select Col 1"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={10}>
        <Form.Item label="Col 2" name="col2">
          <Select
            allowClear
            placeholder="Select Col 2"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
    </Row>
  );

  const TakeFirstN = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="n" name="n">
          <InputNumber placeholder="n" style={{ width: "100%" }} />
        </Form.Item>
      </Col>
    </Row>
  );

  const TakeLastN = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="n" name="n">
          <InputNumber placeholder="n" style={{ width: "100%" }} />
        </Form.Item>
      </Col>
    </Row>
  );

  const CreateTransformedColumn = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={8}>
        <Form.Item label="Transform Type" name="transform">
          <Select
            allowClear
            placeholder="Select Transform Type"
            disabled={!file}
            options={transformTypeData}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="n" name="n">
          <InputNumber placeholder="n" style={{ width: "100%" }} />
        </Form.Item>
      </Col>
    </Row>
  );

  const ReplaceValues = () => (
    <Row gutter={8}>
      <Col span={8}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={8}>
        <Form.Item label="Value Search" name="val_search">
          <Input placeholder="Value Search" />
        </Form.Item>
      </Col>
      <Col span={8}>
        <Form.Item label="Value Replace" name="val_replace">
          <Input placeholder="Value Replace" />
        </Form.Item>
      </Col>
    </Row>
  );

  const ScaleColumnInDataframe = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={12}>
        <Form.Item label="Scaler Type" name="scaler">
          <Select
            allowClear
            placeholder="Select Scaler Type"
            disabled={!file}
            options={scalerTypeData}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
    </Row>
  );

  const CreateFlagColumn = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={12}>
        <Form.Item label="Value Search" name="val_search">
          <Input placeholder="Value Search" />
        </Form.Item>
      </Col>
    </Row>
  );

  const RemoveOutliers = () => (
    <Row gutter={8}>
      <Col span={12}>
        <Form.Item label="Column Name" name="col_name">
          <Select
            allowClear
            placeholder="Select column name"
            disabled={!file}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="Q1" name="Q1">
          <InputNumber placeholder="Q1" style={{ width: "100%" }} />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="Q3" name="Q3">
          <InputNumber placeholder="Q3" style={{ width: "100%" }} />
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item label="Remove" name="remove">
          <Checkbox>Remove</Checkbox>
        </Form.Item>
      </Col>
    </Row>
  );

  return (
    <Col className="feature-engineering-container">
      <Row className="feature-engineering-title">
        <Typography.Title level={4}>Feature Engineering</Typography.Title>
      </Row>
      <Row gutter={16}>
        <Col span={6}>
          <Dragger
            style={{ padding: "0 20px" }}
            accept=".xlsx, .xls, .csv"
            maxCount={1}
            onRemove={(file) => {
              updateFile(null);
            }}
            beforeUpload={(file) => {
              if (isFileSizeWithinLimit(file.size)) {
                updateFile(file);
                const reader = new FileReader();

                reader.onload = ({ target }) => {
                  const data = target?.result?.toString();
                  if (data) {
                    Papa.parse<string[]>(data, {
                      complete: ({ data }) => {
                        updateTargetColumns(data[0]);
                      },
                    });
                  }
                };
                reader.readAsBinaryString(file);
              } else {
                message.error(
                  `File size must be smaller than ${constant.MaxFileSizeMB}MB!`,
                );
              }
              return false;
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag file to this area to upload
            </p>
            <p className="ant-upload-hint">
              Support for a single upload. (CSV, XLSX, XLS) <br />
              Max file size: {constant.MaxFileSizeMB} MB
            </p>
          </Dragger>
        </Col>
        <Col span={18}>
          <Form
            layout="vertical"
            form={form}
            initialValues={{
              operation: operationTypeData[0].value,
              method: null,
              scaler: null,
              col_name: null,
              col1: null,
              col2: null,
              n: null,
              Q1: null,
              Q3: null,
              remove: false,
              val_search: "",
              val_replace: "",
            }}
            onFinish={onFinish}
          >
            <Row gutter={8}>
              <Col span={12}>
                <Form.Item
                  label="Operation"
                  name="operation"
                  rules={[
                    {
                      required: true,
                      message: "Please select an operation!",
                    },
                  ]}
                >
                  <Select
                    placeholder="Select Operation"
                    disabled={!file}
                    options={operationTypeData}
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              </Col>
            </Row>
            {operation === OperationType.PerformOperation ? (
              <PerformOperation />
            ) : operation === OperationType.TakeFirstN ? (
              <TakeFirstN />
            ) : operation === OperationType.TakeLastN ? (
              <TakeLastN />
            ) : operation === OperationType.CreateTransformedColumn ? (
              <CreateTransformedColumn />
            ) : operation === OperationType.ScaleColumnInDataframe ? (
              <ScaleColumnInDataframe />
            ) : operation === OperationType.ReplaceValues ? (
              <ReplaceValues />
            ) : operation === OperationType.CreateFlagColumn ? (
              <CreateFlagColumn />
            ) : (
              operation === OperationType.RemoveOutliers && <RemoveOutliers />
            )}
            <Row justify="end">
              <Button
                type="primary"
                htmlType="submit"
                disabled={!file || loadingFeatureEngineering}
              >
                {loadingFeatureEngineering ? (
                  <>
                    <LoadingOutlined /> Feature Engineering...
                  </>
                ) : (
                  "Run Feature Engineering"
                )}
              </Button>
            </Row>
          </Form>
        </Col>
      </Row>
    </Col>
  );
};

export default FeatureEngineering;
