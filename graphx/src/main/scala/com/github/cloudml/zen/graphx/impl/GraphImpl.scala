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

package com.github.cloudml.zen.graphx.impl

import com.github.cloudml.zen.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.parameterserver.protocol.matrix.{RowData, Column, Row}
import org.parameterserver.protocol.{DoubleArray, IntArray}
import scala.collection.mutable
import scala.reflect.ClassTag
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import com.github.cloudml.zen.graphx.util.{PSUtils => GPSUtils}

class GraphImpl[VD: ClassTag, ED: ClassTag] protected(
  @transient override val vertices: VertexRDD[VD],
  @transient override val edges: RDD[Edge[ED]]) extends Graph[VD, ED] {

  private var batchSize: Int = 1000

  override def updateVertices(data: RDD[(VertexId, VD)]): Unit = {
    vertices.asInstanceOf[VertexRDDImpl[VD]].updateValues(data, batchSize)
  }

  override def map[ED2: ClassTag](
    fn: (PartitionID, Iterator[EdgeTriplet[VD, ED]], VertexCollector[VD, ED]) => Iterator[Edge[ED2]],
    tripletFields: TripletFields): Graph[VD, ED2] = {
    val psClient = vertices.psClient
    val vName = vertices.psName
    val cleanFn = clean(fn)
    val newEdges = triplets(tripletFields).mapPartitionsWithIndex { case (pid, iter) =>
      val edgeContext = new VertexCollectorImpl[VD, ED](psClient, vName)
      cleanFn(pid, iter, edgeContext)
    }
    new GraphImpl[VD, ED2](vertices, newEdges)
  }

  override def foreach(
    fn: (PartitionID, Iterator[EdgeTriplet[VD, ED]], VertexCollector[VD, ED]) => Unit,
    tripletFields: TripletFields): Unit = {
    val psClient = vertices.psClient
    val vName = vertices.psName
    val cleanFn = clean(fn)
    val c = triplets(tripletFields).mapPartitionsWithIndex { case (pid, iter) =>
      val edgeContext = new VertexCollectorImpl[VD, ED](psClient, vName)
      cleanFn(pid, iter, edgeContext)
      Array(1).toIterator
    }
    c.count()
  }

  override def aggregateMessages(
    fn: (EdgeTriplet[VD, ED], VertexCollector[VD, ED]) => Unit,
    tripletFields: TripletFields = TripletFields.All): VertexRDD[VD] = {
    val vi = vertices.asInstanceOf[VertexRDDImpl[VD]]
    val masterSockAddr = vi.masterSockAddr
    val psClient = vi.psClient
    val isDense = vi.isDense
    val rowSize = vi.rowSize
    val colSize = vi.colSize
    val cleanFn = clean(fn)
    val vName = GPSUtils.create[VD](psClient,isDense,rowSize.toInt,colSize.toInt )
    triplets(tripletFields).foreachPartition { iter =>
      val edgeContext = new VertexCollectorImpl[VD, ED](psClient, vName)
      iter.foreach(e => cleanFn(e, edgeContext))
    }
    new VertexRDDImpl[VD](vi.partitionsRDD, masterSockAddr, vName, isDense, rowSize, colSize, vi.targetStorageLevel)
  }

  override def mapVertices[VD2: ClassTag](fn: (VertexId, VD) => VD2): Graph[VD2, ED] = {
    val vi = vertices.asInstanceOf[VertexRDDImpl[VD]]
    val masterSockAddr = vi.masterSockAddr
    val psClient = vi.psClient
    val isDense = vi.isDense
    val rowSize = vi.rowSize
    val colSize = vi.colSize
    val cleanFn = clean(fn)
    val vName = GPSUtils.create[VD2](psClient)
    val newVertices = new VertexRDDImpl[VD2](vi.partitionsRDD, masterSockAddr, vName,
      isDense, rowSize, colSize, vi.targetStorageLevel)
    val data = vertices.map { case (vid, value) =>
      (vid, cleanFn(vid, value))
    }
    newVertices.updateValues(data, batchSize)
    new GraphImpl[VD2, ED](newVertices, edges)
  }

  override def mapEdges[ED2: ClassTag](fn: (PartitionID, Iterator[Edge[ED]]) =>
    Iterator[Edge[ED2]]): Graph[VD, ED2] = {
    val cleanFn = clean(fn)
    val newEdges = edges.mapPartitionsWithIndex { case (pid, iter) =>
      cleanFn(pid, iter)
    }
    new GraphImpl[VD, ED2](vertices, newEdges)
  }

  override def mapTriplets[ED2: ClassTag](
    fn: (PartitionID, Iterator[EdgeTriplet[VD, ED]]) => Iterator[Edge[ED2]],
    tripletFields: TripletFields): Graph[VD, ED2] = {
    val cleanFn = clean(fn)
    val newEdges = triplets(tripletFields).mapPartitionsWithIndex { case (pid, iter) =>
      cleanFn(pid, iter)
    }
    new GraphImpl[VD, ED2](vertices, newEdges)
  }

  override def persist(newLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): this.type = {
    edges.persist(newLevel)
    vertices.persist(newLevel)
    this
  }

  override def unpersist(blocking: Boolean = true): this.type = {
    edges.unpersist(blocking)
    vertices.unpersist(blocking)
    this
  }

  /**
   * TODO:  临时这样问题的.
   * @param blocking
   */

  override def destroy(blocking: Boolean): Unit = {
    edges.unpersist(blocking)
    vertices.unpersist(blocking)
  }

  /**
   * TODO:  临时这样问题的.
   */

  override def checkpoint(): Unit = {
    edges.checkpoint()
    vertices.checkpoint()
  }

  override def isCheckpointed: Boolean = {
    edges.isCheckpointed
  }

  override def getCheckpointFiles: Seq[String] = {
    edges.getCheckpointFile.toSeq
  }

  override def subgraph(
    epred: EdgeTriplet[VD, ED] => Boolean = x => true,
    vpred: (VertexId, VD) => Boolean = (v, d) => true): Graph[VD, ED] = {
    val newEdges = triplets(TripletFields.All).filter(epred).map(e => e.asInstanceOf[Edge[ED]])
    val cleanFn = clean(vpred)
    val v = vertices.filter(v => cleanFn(v._1, v._2))
    val vi = vertices.asInstanceOf[VertexRDDImpl[VD]]
    val masterSockAddr = vi.masterSockAddr
    val psClient = vi.psClient
    val isDense = vi.isDense
    val rowSize = vi.rowSize
    val colSize = vi.colSize
    val storageLevel = vi.targetStorageLevel
    val vName = GPSUtils.create[VD](psClient,isDense,rowSize.toInt,colSize.toInt )
    val newVertices = new VertexRDDImpl[VD](v.map(_._1), masterSockAddr, vName, isDense,
      rowSize, colSize, storageLevel)
    newVertices.updateValues(v)
    new GraphImpl[VD, ED](newVertices, newEdges)
  }

  override def triplets(tripletFields: TripletFields = TripletFields.All): RDD[EdgeTriplet[VD, ED]] = {
    val psClient = vertices.psClient
    val vName = vertices.psName
    edges.mapPartitionsWithIndex { (pid, iter) =>
      if (tripletFields == TripletFields.All ||
        tripletFields == TripletFields.Dst || tripletFields == TripletFields.Src) {
        iter.grouped(batchSize).map { batchEdges =>
          val vidSet = mutable.HashSet[Long]()
          batchEdges.foreach { e =>
            if (tripletFields == TripletFields.All) {
              vidSet.add(e.srcId)
              vidSet.add(e.dstId)
            } else if (tripletFields == TripletFields.Dst) {
              vidSet.add(e.dstId)
            } else {
              vidSet.add(e.srcId)
            }
          }
          val indices = vidSet.toArray
          val v2i = new mutable.OpenHashMap[Long, Int]()
          indices.zipWithIndex.foreach { case (vid, offset) =>
            v2i(vid) = offset
          }
          val vd = GPSUtils.get[VD](psClient,vName,indices.map(_.toInt))
          batchEdges.map { edge =>
            var srcAttr: VD = null.asInstanceOf[VD]
            var dstAttr: VD = null.asInstanceOf[VD]
            if (tripletFields == TripletFields.All) {
              srcAttr = vd(v2i(edge.srcId))
              dstAttr = vd(v2i(edge.dstId))
            } else if (tripletFields == TripletFields.Dst) {
              dstAttr = vd(v2i(edge.dstId))
            } else {
              srcAttr = vd(v2i(edge.srcId))
            }
            edge.toEdgeTriplet(srcAttr, dstAttr)
          }
        }.flatten
      } else {
        iter.map { edge =>
          val srcAttr: VD = null.asInstanceOf[VD]
          val dstAttr: VD = null.asInstanceOf[VD]
          edge.toEdgeTriplet(srcAttr, dstAttr)
        }
      }
    }
  }
}
