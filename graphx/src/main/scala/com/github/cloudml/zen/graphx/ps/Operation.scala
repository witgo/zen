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

package com.github.cloudml.zen.graphx.ps

import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

import java.util.UUID

import breeze.linalg.{DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import breeze.storage.Zero
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.parameterserver.client.PSClient
import org.parameterserver.protocol.matrix.{Row, RowData}
import org.parameterserver.protocol.vector.{DenseVector => PDV, SparseVector => PSV, Vector => PV}
import org.parameterserver.protocol.{Array => PSArray, FloatArray, DataType, DoubleArray, IntArray, LongArray}

trait Operation[VD] extends Serializable {
  def dataType: DataType

  def remove(psClient: PSClient, psName: String): Unit

  def create(psClient: PSClient, dense: Boolean, rowNum: Int, columnNum: Int): String

  def update(psClient: PSClient, psName: String, indices: Array[Int], values: Array[VD]): Unit

  def inc(psClient: PSClient, psName: String, indices: Array[Int], values: Array[VD]): Unit

  def get(psClient: PSClient, psName: String, indices: Array[Int]): Array[VD]
}

private[graphx] abstract class VectorOperation[VD] extends Operation[VD] {
  // implicit protected def vdTag: ClassTag[VD] = implicitly[ClassTag[VD]]

  def javaArray2RowData(indices: Array[Int], values: Array[VD]): Array[RowData]

  def rowData2JavaArray(values: Array[RowData]): Array[VD]

  override def remove(psClient: PSClient, psName: String): Unit = {
    psClient.removeMatrix(psName)
  }

  override def create(
    psClient: PSClient,
    dense: Boolean = false,
    rowNum: Int = Int.MaxValue,
    columnNum: Int = Int.MaxValue): String = {
    val name = UUID.randomUUID().toString
    psClient.createMatrix(name, false, dense, rowNum, columnNum, dataType)
    name
  }

  override def update(psClient: PSClient, psName: String,
    indices: Array[Int], values: Array[VD]): Unit = {
    psClient.updateMatrix(psName, javaArray2RowData(indices, values))
  }

  override def inc(psClient: PSClient, psName: String,
    indices: Array[Int], values: Array[VD]): Unit = {
    psClient.add2Matrix(psName, javaArray2RowData(indices, values))
  }

  override def get(psClient: PSClient, psName: String, indices: Array[Int]): Array[VD] = {
    val rows = indices.map(i => new Row(i))
    val data = psClient.getMatrix(psName, rows)
    rowData2JavaArray(data)
  }
}

private[graphx] abstract class BVOperation[V: ClassTag : Zero] extends VectorOperation[BV[V]] {
  def javaArray2PsArray(values: Array[V]): PSArray[_]

  def psArray2JavaArray(values: PSArray[_]): Array[V]

  override def javaArray2RowData(indices: Array[Int], values: Array[BV[V]]): Array[RowData] = {
    val rowDatas = new Array[RowData](indices.length)
    for (i <- indices.indices) {
      val r = indices(i)
      val bv = values(i)
      val rowData = new RowData(r)
      val pSize = bv.size
      val pIndices = bv.activeKeysIterator.toArray
      val pData = bv.activeValuesIterator.toArray
      rowData.setData(new PSV(pSize, pIndices, javaArray2PsArray(pData)))
      rowDatas(i) = rowData
    }
    rowDatas
  }

  override def rowData2JavaArray(data: Array[RowData]): Array[BV[V]] = {
    val rowDatas = new Array[BV[V]](data.length)
    for (i <- data.indices) {
      val rd = data(i)
      val pv = rd.getData
      rowDatas(i) = if (pv == null || pv.getValues == null) {
        BSV.zeros[V](Int.MaxValue)
      } else if (pv.isInstanceOf[PDV]) {
        val pdv = pv.asInstanceOf[PDV]
        BDV[V](psArray2JavaArray(pdv.getValues))
      } else {
        val psv = pv.asInstanceOf[PSV]
        new BSV[V](psv.getIndices, psArray2JavaArray(psv.getValues), psv.getSize)
      }
    }
    rowDatas
  }
}

private[graphx] abstract class BSVOperation[V: ClassTag : Zero] extends VectorOperation[BSV[V]] {
  def javaArray2PsArray(values: Array[V]): PSArray[_]

  def psArray2JavaArray(values: PSArray[_]): Array[V]

  override def javaArray2RowData(indices: Array[Int], values: Array[BSV[V]]): Array[RowData] = {
    val rowDatas = new Array[RowData](indices.length)
    for (i <- indices.indices) {
      val r = indices(i)
      val bv = values(i)
      val rowData = new RowData(r)
      val pSize = bv.size
      val pIndices = bv.activeKeysIterator.toArray
      val pData = bv.activeValuesIterator.toArray
      rowData.setData(new PSV(pSize, pIndices, javaArray2PsArray(pData)))
      rowDatas(i) = rowData
    }
    rowDatas
  }

  override def rowData2JavaArray(data: Array[RowData]): Array[BSV[V]] = {
    val rowDatas = new Array[BSV[V]](data.length)
    for (i <- data.indices) {
      val rd = data(i)
      val pv = rd.getData
      rowDatas(i) = if (pv == null || pv.getValues == null) {
        BSV.zeros[V](Int.MaxValue)
      } else if (pv.isInstanceOf[PDV]) {
        val pdv = pv.asInstanceOf[PDV]
        new BSV[V]((0 until pdv.getSize).toArray, psArray2JavaArray(pdv.getValues), pdv.getSize)
      } else {
        val psv = pv.asInstanceOf[PSV]
        new BSV[V](psv.getIndices, psArray2JavaArray(psv.getValues), psv.getSize)
      }
    }
    rowDatas
  }
}

private[graphx] abstract class BDVOperation[V: ClassTag : Zero] extends VectorOperation[BDV[V]] {
  def javaArray2PsArray(values: Array[V]): PSArray[_]

  def psArray2JavaArray(values: PSArray[_]): Array[V]

  override def javaArray2RowData(indices: Array[Int], values: Array[BDV[V]]): Array[RowData] = {
    val rowDatas = new Array[RowData](indices.length)
    for (i <- indices.indices) {
      val r = indices(i)
      val bv = values(i)
      val rowData = new RowData(r)
      val pSize = bv.size
      val pIndices = bv.activeKeysIterator.toArray
      val pData = bv.activeValuesIterator.toArray
      rowData.setData(new PSV(pSize, pIndices, javaArray2PsArray(pData)))
      rowDatas(i) = rowData
    }
    rowDatas
  }

  override def rowData2JavaArray(data: Array[RowData]): Array[BDV[V]] = {
    val rowDatas = new Array[BDV[V]](data.length)
    for (i <- data.indices) {
      val rd = data(i)
      val pv = rd.getData
      rowDatas(i) = if (pv == null || pv.getValues == null) {
        BDV.zeros[V](Int.MaxValue)
      } else {
        val pdv = pv.asInstanceOf[PDV]
        BDV[V](psArray2JavaArray(pdv.getValues))
      }
    }
    rowDatas
  }
}

private[graphx] abstract class ValOperation[VD: ClassTag] extends Operation[VD] {

  protected def vdTag: ClassTag[VD] = implicitly[ClassTag[VD]]

  def javaArray2PsArray(values: Array[VD]): PSArray[_]

  def psArray2JavaArray(values: PSArray[_]): Array[VD]

  override def remove(psClient: PSClient, psName: String): Unit = {
    psClient.removeVector(psName)
  }

  override def create(
    psClient: PSClient,
    dense: Boolean = false,
    rowNum: Int = Int.MaxValue,
    columnNum: Int = Int.MaxValue): String = {
    val name = UUID.randomUUID().toString
    psClient.createVector(name, dense, rowNum, dataType)
    name
  }

  override def update(psClient: PSClient, psName: String,
    indices: Array[Int], values: Array[VD]): Unit = {
    psClient.updateVector(psName, indices, javaArray2PsArray(values))
  }

  override def inc(psClient: PSClient, psName: String,
    indices: Array[Int], values: Array[VD]): Unit = {
    psClient.add2Vector(psName, indices, javaArray2PsArray(values))
  }

  override def get(psClient: PSClient, psName: String, indices: Array[Int]): Array[VD] = {
    val psArray = psClient.getVector(psName, indices)
    psArray2JavaArray(psArray)
  }
}

object Operation {

  object IntegerOperation extends ValOperation[Int] {

    override def dataType: DataType = DataType.Integer

    override def javaArray2PsArray(values: Array[Int]): PSArray[_] = {
      new IntArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Int] = {
      values.asInstanceOf[IntArray].getValues
    }
  }

  object DoubleOperation extends ValOperation[Double] {

    override def dataType: DataType = DataType.Double

    override def javaArray2PsArray(values: Array[Double]): PSArray[_] = {
      new DoubleArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Double] = {
      values.asInstanceOf[DoubleArray].getValues
    }
  }

  object FloatOperation extends ValOperation[Float] {

    override def dataType: DataType = DataType.Float

    override def javaArray2PsArray(values: Array[Float]): PSArray[_] = {
      new FloatArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Float] = {
      values.asInstanceOf[FloatArray].getValues
    }
  }

  object LongOperation extends ValOperation[Long] {

    override def dataType: DataType = DataType.Long

    override def javaArray2PsArray(values: Array[Long]): PSArray[_] = {
      new LongArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Long] = {
      values.asInstanceOf[LongArray].getValues
    }
  }

  object IntegerBVOperation extends BVOperation[Int] {

    override def dataType: DataType = DataType.Integer

    override def javaArray2PsArray(values: Array[Int]): PSArray[_] = {
      new IntArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Int] = {
      values.asInstanceOf[IntArray].getValues
    }
  }

  object DoubleBVOperation extends BVOperation[Double] {

    override def dataType: DataType = DataType.Double

    override def javaArray2PsArray(values: Array[Double]): PSArray[_] = {
      new DoubleArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Double] = {
      values.asInstanceOf[DoubleArray].getValues
    }
  }

  object FloatBVOperation extends BVOperation[Float] {

    override def dataType: DataType = DataType.Float

    override def javaArray2PsArray(values: Array[Float]): PSArray[_] = {
      new FloatArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Float] = {
      values.asInstanceOf[FloatArray].getValues
    }
  }

  object LongBVOperation extends BVOperation[Long] {

    override def dataType: DataType = DataType.Long

    override def javaArray2PsArray(values: Array[Long]): PSArray[_] = {
      new LongArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Long] = {
      values.asInstanceOf[LongArray].getValues
    }
  }

  object IntegerBSVOperation extends BSVOperation[Int] {

    override def dataType: DataType = DataType.Integer

    override def javaArray2PsArray(values: Array[Int]): PSArray[_] = {
      new IntArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Int] = {
      values.asInstanceOf[IntArray].getValues
    }
  }

  object DoubleBSVOperation extends BSVOperation[Double] {

    override def dataType: DataType = DataType.Double

    override def javaArray2PsArray(values: Array[Double]): PSArray[_] = {
      new DoubleArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Double] = {
      values.asInstanceOf[DoubleArray].getValues
    }
  }

  object FloatBSVOperation extends BSVOperation[Float] {

    override def dataType: DataType = DataType.Float

    override def javaArray2PsArray(values: Array[Float]): PSArray[_] = {
      new FloatArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Float] = {
      values.asInstanceOf[FloatArray].getValues
    }
  }

  object LongBSVOperation extends BSVOperation[Long] {

    override def dataType: DataType = DataType.Long

    override def javaArray2PsArray(values: Array[Long]): PSArray[_] = {
      new LongArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Long] = {
      values.asInstanceOf[LongArray].getValues
    }
  }

  object IntegerBDVOperation extends BDVOperation[Int] {

    override def dataType: DataType = DataType.Integer

    override def javaArray2PsArray(values: Array[Int]): PSArray[_] = {
      new IntArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Int] = {
      values.asInstanceOf[IntArray].getValues
    }
  }

  object DoubleBDVOperation extends BDVOperation[Double] {

    override def dataType: DataType = DataType.Double

    override def javaArray2PsArray(values: Array[Double]): PSArray[_] = {
      new DoubleArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Double] = {
      values.asInstanceOf[DoubleArray].getValues
    }
  }

  object FloatBDVOperation extends BDVOperation[Float] {

    override def dataType: DataType = DataType.Float

    override def javaArray2PsArray(values: Array[Float]): PSArray[_] = {
      new FloatArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Float] = {
      values.asInstanceOf[FloatArray].getValues
    }
  }

  object LongBDVOperation extends BDVOperation[Long] {

    override def dataType: DataType = DataType.Long

    override def javaArray2PsArray(values: Array[Long]): PSArray[_] = {
      new LongArray(values)
    }

    override def psArray2JavaArray(values: PSArray[_]): Array[Long] = {
      values.asInstanceOf[LongArray].getValues
    }
  }

  object SVOperation extends VectorOperation[SV] {
    override def dataType: DataType = DataType.Double

    override def javaArray2RowData(indices: Array[Int], values: Array[SV]): Array[RowData] = {
      val rowDatas = new Array[RowData](indices.length)
      for (i <- indices.indices) {
        val r = indices(i)
        val v = values(i)
        val rowData = new RowData(r)
        v match {
          case SDV(values) =>
            rowData.setData(new PSV(values.length, (0 until values.length).toArray, new DoubleArray(values)))
          case SSV(size, indices, values) =>
            rowData.setData(new PSV(size, indices, new DoubleArray(values)))
        }
        rowDatas(i) = rowData
      }
      rowDatas
    }

    override def rowData2JavaArray(data: Array[RowData]): Array[SV] = {
      val rowDatas = new Array[SV](data.length)
      for (i <- data.indices) {
        val rd = data(i)
        val pv = rd.getData
        rowDatas(i) = if (pv == null || pv.getValues == null) {
          new SSV(Int.MaxValue, Array.empty[Int], Array.empty[Double])
        } else if (pv.isInstanceOf[PDV]) {
          val pdv = pv.asInstanceOf[PDV]
          new SDV(pdv.getValues.asInstanceOf[DoubleArray].getValues)
        } else {
          val psv = pv.asInstanceOf[PSV]
          new SSV(psv.getSize, psv.getIndices, psv.getValues.asInstanceOf[DoubleArray].getValues)
        }
      }
      rowDatas
    }
  }

  object SSVOperation extends VectorOperation[SSV] {
    override def dataType: DataType = DataType.Double

    override def javaArray2RowData(indices: Array[Int], values: Array[SSV]): Array[RowData] = {
      val rowDatas = new Array[RowData](indices.length)
      for (i <- indices.indices) {
        val r = indices(i)
        val v = values(i)
        val rowData = new RowData(r)
        val SSV(psSize, psIndices, psValues) = v
        rowData.setData(new PSV(psSize, psIndices, new DoubleArray(psValues)))
        rowDatas(i) = rowData
      }
      rowDatas
    }

    override def rowData2JavaArray(data: Array[RowData]): Array[SSV] = {
      val rowDatas = new Array[SSV](data.length)
      for (i <- data.indices) {
        val rd = data(i)
        val pv = rd.getData
        rowDatas(i) = if (pv == null || pv.getValues == null) {
          new SSV(Int.MaxValue, Array.empty[Int], Array.empty[Double])
        } else {
          val psv = pv.asInstanceOf[PSV]
          new SSV(psv.getSize, psv.getIndices, psv.getValues.asInstanceOf[DoubleArray].getValues)
        }
      }
      rowDatas
    }
  }

  object SDVOperation extends VectorOperation[SDV] {
    override def dataType: DataType = DataType.Double

    override def javaArray2RowData(indices: Array[Int], values: Array[SDV]): Array[RowData] = {
      val rowDatas = new Array[RowData](indices.length)
      for (i <- indices.indices) {
        val r = indices(i)
        val v = values(i)
        val rowData = new RowData(r)
        rowData.setData(new PDV(new DoubleArray(v.toArray)))
        rowDatas(i) = rowData
      }
      rowDatas
    }

    override def rowData2JavaArray(data: Array[RowData]): Array[SDV] = {
      val rowDatas = new Array[SDV](data.length)
      for (i <- data.indices) {
        val rd = data(i)
        val pv = rd.getData
        rowDatas(i) = new SDV(pv.asInstanceOf[PDV].getValues.asInstanceOf[DoubleArray].getValues)
      }
      rowDatas
    }
  }

  lazy val class2Operatoin: mutable.OpenHashMap[Class[_], Operation[_]] = {
    val hash = new mutable.OpenHashMap[Class[_], Operation[_]]()
    hash.put(classOf[Int], IntegerOperation)
    hash.put(classOf[Float], FloatOperation)
    hash.put(classOf[Double], DoubleOperation)
    hash.put(classOf[Long], LongOperation)

    hash.put(classOf[SV], SVOperation)
    hash.put(classOf[SSV], SSVOperation)
    hash.put(classOf[SDV], SDVOperation)

    hash.put(classOf[BV[Int]], IntegerBVOperation)
    hash.put(classOf[BV[Float]], FloatBVOperation)
    hash.put(classOf[BV[Double]], DoubleBVOperation)
    hash.put(classOf[BV[Long]], LongBVOperation)

    hash.put(classOf[BSV[Int]], IntegerBSVOperation)
    hash.put(classOf[BSV[Float]], FloatBSVOperation)
    hash.put(classOf[BSV[Double]], DoubleBSVOperation)
    hash.put(classOf[BSV[Long]], LongBSVOperation)

    hash.put(classOf[BDV[Int]], IntegerBDVOperation)
    hash.put(classOf[BDV[Float]], FloatBDVOperation)
    hash.put(classOf[BDV[Double]], DoubleBDVOperation)
    hash.put(classOf[BDV[Long]], LongBDVOperation)
    hash
  }

  def remove[VD](psClient: PSClient, psName: String)(implicit tag: ClassTag[VD]): Unit = {
    tag2Operation(tag).remove(psClient, psName)
  }

  def create[VD](psClient: PSClient, dense: Boolean, rowNum: Int, columnNum: Int)
    (implicit tag: ClassTag[VD]): String = {
    tag2Operation(tag).create(psClient, dense, rowNum, columnNum)
  }

  def update[VD](psClient: PSClient, psName: String, indices: Array[Int], values: Array[VD])
    (implicit tag: ClassTag[VD]): Unit = {
    tag2Operation(tag).update(psClient, psName, indices, values)
  }

  def inc[VD](psClient: PSClient, psName: String, indices: Array[Int], values: Array[VD])
    (implicit tag: ClassTag[VD]): Unit = {
    tag2Operation(tag).inc(psClient, psName, indices, values)
  }

  def get[VD](psClient: PSClient, psName: String, indices: Array[Int])
    (implicit tag: ClassTag[VD]): Array[VD] = {
    tag2Operation(tag).get(psClient, psName, indices)
  }

  private def tag2Operation[VD](tag: ClassTag[VD]): Operation[VD] = {
    val vdClass = tag.runtimeClass
    if (class2Operatoin.contains(vdClass)) {
      class2Operatoin(vdClass).asInstanceOf[Operation[VD]]
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }
}
