/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.graphx.util

import java.util.UUID

import org.parameterserver.client.PSClient
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import com.github.cloudml.zen.graphx._
import org.parameterserver.protocol.matrix.{RowData, Column, Row}
import org.parameterserver.protocol.{DataType, DoubleArray, IntArray}
import scala.reflect.ClassTag
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV,
DenseMatrix => SDM, SparseMatrix => SSM, Matrix => SM}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV,
CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}

private[graphx] object PSUtils {

  def remove[VD: ClassTag](psClient: PSClient, psName: String): Unit = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (classOf[SV].isAssignableFrom(vdClass)) {
      psClient.removeMatrix(psName)
    } else {
      psClient.removeVector(psName)
    }
  }

  def create[VD: ClassTag](
    psClient: PSClient,
    dense: Boolean = false,
    rowNum: Int = Int.MaxValue,
    columnNum: Int = Int.MaxValue): String = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    val name = UUID.randomUUID().toString
    if (classOf[SV].isAssignableFrom(vdClass)) {
      psClient.createMatrix(name, dense, rowNum, columnNum, DataType.Double)
    } else if (vdClass == java.lang.Integer.TYPE) {
      psClient.createVector(name, dense, rowNum, DataType.Integer)
    } else if (vdClass == java.lang.Double.TYPE) {
      psClient.createVector(name, dense, rowNum, DataType.Double)
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
    name
  }

  def update[VD: ClassTag](psClient: PSClient, psName: String, value: VD): Unit = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (classOf[SV].isAssignableFrom(vdClass)) {
      val (indices, array) = sv2DoubleArray(value.asInstanceOf[SV])
      if (indices == null) {
        psClient.updateVector(psName, array)
      } else {
        psClient.updateVector(psName, indices, array)
      }
    } else if (classOf[SM].isAssignableFrom(vdClass)) {
      val rowDatas = sm2RowData(value.asInstanceOf[SM])
      psClient.updateMatrix(psName, rowDatas)
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  def batchUpadte[VD: ClassTag](psClient: PSClient, psName: String, indices: Array[Int], values: Array[VD]): Unit = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (vdClass == java.lang.Integer.TYPE) {
      psClient.updateVector(psName, indices, new IntArray(values.asInstanceOf[Array[Int]]))
    } else if (vdClass == java.lang.Double.TYPE) {
      psClient.updateVector(psName, indices, new DoubleArray(values.asInstanceOf[Array[Double]]))
    } else if (classOf[SV].isAssignableFrom(vdClass)) {
      val data = sv2RowData(indices, values.map(_.asInstanceOf[SV]))
      psClient.updateMatrix(psName, data)
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  def inc[VD: ClassTag](psClient: PSClient, psName: String, value: VD): Unit = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (classOf[SV].isAssignableFrom(vdClass)) {
      val (indices, array) = sv2DoubleArray(value.asInstanceOf[SV])
      if (indices == null) {
        psClient.updateVector(psName, array)
      } else {
        psClient.updateVector(psName, indices, array)
      }
    } else if (classOf[SM].isAssignableFrom(vdClass)) {
      val rowDatas = sm2RowData(value.asInstanceOf[SM])
      psClient.updateMatrix(psName, rowDatas)
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  def batchInc[VD: ClassTag](psClient: PSClient, psName: String,
    indices: Array[Int], values: Array[VD]): Unit = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (vdClass == java.lang.Integer.TYPE) {
      psClient.add2Vector(psName, indices, new IntArray(values.asInstanceOf[Array[Int]]))
    } else if (vdClass == java.lang.Double.TYPE) {
      psClient.add2Vector(psName, indices, new DoubleArray(values.asInstanceOf[Array[Double]]))
    } else if (classOf[SV].isAssignableFrom(vdClass)) {
      val data = sv2RowData(indices, values.map(_.asInstanceOf[SV]))
      psClient.add2Matrix(psName, data)
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  def get[VD: ClassTag](psClient: PSClient, psName: String,
    indices: Array[Int]): Array[VD] = {
    val vdClass = implicitly[ClassTag[VD]].runtimeClass
    if (vdClass == java.lang.Integer.TYPE) {
      psClient.getVector(psName, indices).asInstanceOf[IntArray].getValues.asInstanceOf[Array[VD]]
    } else if (vdClass == java.lang.Double.TYPE) {
      psClient.getVector(psName, indices).asInstanceOf[DoubleArray].getValues.asInstanceOf[Array[VD]]
    } else if (classOf[SV].isAssignableFrom(vdClass)) {
      val rows = indices.map(i => new Row(i))
      val data = psClient.getMatrix(psName, rows)
      rowData2sv(data).asInstanceOf[Array[VD]]
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  private def sv2DoubleArray(value: SV): (Array[Int], DoubleArray) = {
    value match {
      case SDV(values) =>
        (null, new DoubleArray(values))
      case SSV(size, indices, values) =>
        (indices, new DoubleArray(values))
    }
  }

  private def sv2RowData(indices: Array[Int], values: Array[SV]): Array[RowData] = {
    val rowDatas = new Array[RowData](indices.length)
    for (i <- indices.indices) {
      val r = indices(i)
      val v = values(i)
      val rowData = new RowData(r)
      v match {
        case SDV(values) =>
          rowData.setData(new DoubleArray(values))
          rowData.setColumns(values.indices.toArray)
        case SSV(size, indices, values) =>
          rowData.setData(new DoubleArray(values))
          rowData.setColumns(indices)
      }
      rowDatas(i) = rowData
    }
    rowDatas
  }

  private def rowData2sv(data: Array[RowData]): Array[SV] = {
    val rowDatas = new Array[SV](data.length)
    for (i <- data.indices) {
      val rd = data(i)
      rowDatas(i) = if (rd.getData == null) {
        new SSV(Int.MaxValue, Array.empty[Int], Array.empty[Double])
      } else if (rd.getColumns == null) {
        new SDV(rd.getData.asInstanceOf[DoubleArray].getValues)
      } else {
        new SSV(Int.MaxValue, rd.getColumns, rd.getData.asInstanceOf[DoubleArray].getValues)
      }
    }
    rowDatas
  }

  private def sm2RowData(value: SM): Array[RowData] = {
    if (value.isInstanceOf[SDM]) {
      val dm = value.asInstanceOf[SDM]
      val bdm = if (!dm.isTransposed) {
        new BDM[Double](dm.numRows, dm.numCols, dm.values)
      } else {
        val breezeMatrix = new BDM[Double](dm.numCols, dm.numRows, dm.values)
        breezeMatrix.t
      }
      val values = new Array[RowData](bdm.rows)
      values.indices.foreach { i =>
        val sv = bdm(i, ::)
        val rowData = new RowData(i)
        rowData.setData(new DoubleArray(sv.inner.toArray))
        rowData.setColumns((0 until sv.inner.length).toArray)
        values(i) = rowData
      }
      values
    } else {
      val sm = value.asInstanceOf[SSM]
      val bsm = if (!sm.isTransposed) {
        new BSM[Double](sm.values, sm.numRows, sm.numCols, sm.colPtrs, sm.rowIndices)
      } else {
        val breezeMatrix = new BSM[Double](sm.values, sm.numCols, sm.numRows, sm.colPtrs, sm.rowIndices)
        breezeMatrix.t
      }
      val buff = new Array[ArrayBuffer[(Int, Double)]](bsm.rows)
      bsm.activeIterator.foreach { case ((r, c), v) =>
        if (buff(r) == null) buff(r) = new ArrayBuffer[(Int, Double)]()
        val b = buff(r)
        b.append((c, v))
      }
      buff.zipWithIndex.filter(_._1.nonEmpty).map { case (b, i) =>
        val rowData = new RowData(i)
        val d = b.toArray.unzip
        rowData.setData(new DoubleArray(d._2.toArray))
        rowData.setColumns(d._1.toArray)
        rowData
      }
    }
  }

}
